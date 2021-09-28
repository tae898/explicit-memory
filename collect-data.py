"""Collect data from ConceptNet.
Here we only use the relation `/r/AtLocation`
"""
import requests
from tqdm import tqdm
import logging
from utils import write_json, read_json
from collections import Counter
import random
import argparse
import csv


# for reproducibility
random.seed(42)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def read_mscoco(path: str = './data/ms-coco-80-categories') -> list:
    """Return ms coco 80 object categories."""
    logging.debug(f"Reading {path} ...")
    with open(path, 'r') as stream:
        mscoco = stream.readlines()
    mscoco = [line.strip() for line in mscoco]
    mscoco = ['_'.join(foo.split()) for foo in mscoco]
    logging.info(
        f"Reading {path} complete! There are {len(mscoco)} object categories")

    return mscoco


def read_names(path: str = './data/top-human-names') -> list:
    """Read 20 names."""
    logging.debug(f"Reading {path} ...")
    with open(path, 'r') as stream:
        names = stream.readlines()
    names = [line.strip() for line in names]
    logging.info(
        f"Reading {path} complete! There are {len(names)} object categories")

    return names


def get_from_conceptnet(relation: str, save_at: str) -> None:
    logging.debug(f"retrieving data from conceptnet ...")
    mscoco = read_mscoco()

    data = {}

    for object_category in tqdm(mscoco):
        data[object_category] = []
        query = f'http://api.conceptnet.io/query?start=/c/en/{object_category}&rel={relation}'
        logging.debug(f"making an HTTP get request with query {query}")
        response = requests.get(query).json()

        logging.info(f"{len(response['edges'])} tails (entities) found!")
        for edge in tqdm(response['edges']):
            data[object_category].append(
                {'start': edge['start'],
                 'end': edge['end'],
                 'weight': edge['weight'],
                 'surfaceText': edge['surfaceText']})

    write_json(data, save_at)
    logging.info(f"conceptnet data retrieval done and saved at {save_at}")


def get_conceptnet_data_stats(load_at: str, save_at: str) -> None:
    """Get basic data statistics."""
    logging.debug(f"getting basic data stats from {load_at} ...")
    data = read_json(load_at)
    stats = {'num_examples': {}}
    for key, val in data.items():
        num_examples = len(val)
        stats['num_examples'][key] = num_examples

    all_locations = [
        val_['end']['@id'].split('/')[-1] for key, val in data.items() for val_ in val]
    stats['num_locations_total'] = len(all_locations)
    stats['num_unique_locations'] = len(set(all_locations))

    counts = dict(Counter(all_locations))
    counts = {key: val for key, val in sorted(
        counts.items(), key=lambda x: x[1], reverse=True)}

    stats['location_counts'] = counts

    write_json(stats, save_at)
    logging.info(f"conceptnet data stats saved at {save_at}")


def create_semantic_knowledge(load_at: str, save_at: str, relation: str) -> None:
    """Create semantic (factual) knowledge from the conceptnet data."""
    data = read_json(load_at)

    semantic_knowledge = {}
    for key, val in data.items():
        head = key
        relation = relation.split('/')[-1]
        semantic_knowledge[head] = {relation: []}

        for val_ in val:
            tail = val_['end']['@id'].split('/')[-1]
            semantic_knowledge[head][relation].append(tail)

    write_json(semantic_knowledge, save_at)


def create_semantic_observations(load_at: str, save_at: str, weighting: bool,
                                 num_repeat: int, relation: str) -> None:
    """Create dummy semantic observations using 20 human names.

    10 male and 10 female names were used from here
    https://www.ssa.gov/oact/babynames/decades/century.html
    """
    logging.debug(f"Creating dummy semantic observations ...")

    data = read_json(load_at)
    data_weighted = []

    for key, val in data.items():
        head = key
        relation = relation.split('/')[-1]

        for val_ in val:
            tail = val_['end']['@id'].split('/')[-1]
            weight = round(val_['weight'])

            if not weighting:
                weight = 1
            for _ in range(weight):
                data_weighted.append((key, relation, tail))

    data_weighted = sorted(data_weighted)
    data_weighted = data_weighted * num_repeat

    random.shuffle(data_weighted)

    names = read_names()

    # Note that the observations are all semantic.
    observations = [(random.choice(names) + "'s " + head, edge, tail)
                    for head, edge, tail in data_weighted]

    with open(save_at, 'w') as stream:
        for obs in observations:
            stream.write(f"{obs[0]},{obs[1]},{obs[2]}")
            stream.write('\n')


def create_episodic_observations(load_at: str, save_at: str, episodic_factor: int) -> None:
    """Create episodic observations based on semantic observations."""

    semantic_knowledge = read_json('./data/semantic-knowledge.json')
    possible_locations = list(set(
        [loc for key, val in semantic_knowledge.items() for loc in val['AtLocation']]))
    possible_locations = sorted(possible_locations)
    observations_episodic = []

    observations_semantic = []
    with open(load_at, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            observations_semantic.append(row)

    for obs_s in observations_semantic:
        head = obs_s[0].split()[-1]
        assert obs_s[1] == 'AtLocation'
        tail = obs_s[2]

        for _ in range(episodic_factor):
            while True:
                location_random = random.choice(possible_locations)
                if location_random not in semantic_knowledge[head]['AtLocation']:
                    break
            observations_episodic.append(
                (obs_s[0], 'AtLocation', location_random))

    with open(save_at, 'w') as stream:
        for obs in observations_episodic:
            stream.write(f"{obs[0]},{obs[1]},{obs[2]}")
            stream.write('\n')


def create_splits(s_load_at: str, e_load_at: str,
                  val_ratio: float, test_ratio: float) -> None:
    
    observations_semantic = []
    with open('./data/observations-semantic.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            observations_semantic.append(row)

    observations_episodic = []
    with open('./data/observations-episodic.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            observations_episodic.append(row)


def main(relation: str, save_conceptnet_at: str, save_conceptnet_stats_at: str,
         save_semantic_knowledge_at: str, save_semantic_observations_at: str,
         weighting: bool, num_repeat: bool, episodic_factor: int,
         save_episodic_observations_at: str, val_ratio: float, test_ratio: float) -> None:
    """Collect data."""
    logging.debug(f"Starting collecting data ...")
    # get_from_conceptnet(relation, save_conceptnet_at)
    # get_conceptnet_data_stats(save_conceptnet_at, save_conceptnet_stats_at)
    create_semantic_knowledge(
        save_conceptnet_at, save_semantic_knowledge_at, relation)
    create_semantic_observations(
        save_conceptnet_at, save_semantic_observations_at, weighting, num_repeat, relation)
    create_episodic_observations(
        save_semantic_observations_at, save_episodic_observations_at, episodic_factor)

    create_splits(save_semantic_observations_at,
                  save_episodic_observations_at, val_ratio, test_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data collection')
    parser.add_argument('--relation', default='/r/AtLocation',
                        type=str, help='relation type from conceptnet')
    parser.add_argument('--save-conceptnet-at', default='./data/conceptnet-data.json',
                        type=str, help='where to save raw conceptnet data')
    parser.add_argument('--save-conceptnet-stats-at', default='./data/conceptnet-data-stats.json',
                        type=str, help='where to save raw conceptnet data stats')
    parser.add_argument('--save-semantic-knowledge-at', default='./data/semantic-knowledge.json',
                        type=str, help='where to save semantic knowledge')
    parser.add_argument('--save-semantic-observations-at', default='./data/observations-semantic.csv',
                        type=str, help='where to save dummy semantic observations')
    parser.add_argument('--weighting', default=True, action='store_true',
                        help='whether to sample by weights')
    parser.add_argument('--num-repeat', default=2, type=int,
                        help='number of repeats to make semantic observations bigger.')
    parser.add_argument('--episodic-factor', default=1, type=int,
                        help='number of repeats to make data bigger.')
    parser.add_argument('--save-episodic-observations-at', default='./data/observations-episodic.csv',
                        type=str, help='where to save dummy episodic observations')
    parser.add_argument('--val-ratio', default=0.1,
                        type=float, help='validation split ratio')
    parser.add_argument('--test-ratio', default=0.1,
                        type=float, help='test split ratio')
    args = vars(parser.parse_args())
    print("Arguments:")
    for k, v in args.items():
        print(f"  {k:>21} : {v}")

    main(**args)
