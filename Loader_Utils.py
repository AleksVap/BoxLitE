import os

from DL_Lite_Model.Atomic_Concept import Atomic_Concept
from DL_Lite_Model.Atomic_Role import Atomic_Role

concept_role_translations = {
    '<https://www.wikidata.org/prop/P40>': 'hasChild',
    '<https://www.wikidata.org/prop/P1038>': 'relative',
    '<https://www.wikidata.org/prop/P26>': 'spouse',
    '<https://www.wikidata.org/prop/P8810>': 'hasParent',
    '<https://www.wikidata.org/prop/P22>': 'hasFather',
    '<https://www.wikidata.org/prop/P25>': 'hasMother',
    '<https://www.wikidata.org/prop/P3373>': 'hasSibling',
}

extended_role_inclusions = {
    'relative': {'sup_role': [],
                 'sup_inv_role': ['relative']},
    'spouse': {'sup_role': ['relative'],
               'sup_inv_role': ['spouse', 'relative']},
    'hasSibling': {'sup_role': ['relative'],
                   'sup_inv_role': ['hasSibling', 'relative']},
    'hasChild': {'sup_role': ['relative'],
                 'sup_inv_role': ['hasParent', 'relative']},
    'hasParent': {'sup_role': ['relative'],
                  'sup_inv_role': ['hasChild', 'relative']},
    'hasFather': {'sup_role': ['hasParent', 'relative'],
                  'sup_inv_role': ['hasChild', 'relative']},
    'hasMother': {'sup_role': ['hasParent', 'relative'],
                  'sup_inv_role': ['hasChild', 'relative']},
}


def open_dataset(dataset, mode='train'):
    file = open(os.path.join('Family_Dataset', mode, str(dataset) + '.tsv'), 'r')
    return file


def loadIndividuals(dataset):
    file = open_dataset(dataset, mode='train')
    individuals = set()

    for role_assertion in file:
        values = role_assertion.split()
        individuals.add(values[0])
        individuals.add(values[2])
    individuals = sorted(list(individuals))
    return individuals


def loadDataFromFile(dataset, mode='train'):
    file = open_dataset(dataset, mode=mode)

    data = {
        'concepts': set(),
        'roles': set(),
    }

    for role_assertion in file:
        values = role_assertion.split()
        data['roles'].add((values[0], Atomic_Role(concept_role_translations[values[1]]), values[2]))

    data['concepts'] = sorted(list(data['concepts']), key=lambda x: x[1].get_op_name() + x[0], )
    data['roles'] = sorted(list(data['roles']), key=lambda x: x[1].get_op_name() + x[0] + x[2])
    return data


def loadFilterAssertions(dataset):
    file = open_dataset(dataset, mode='train')

    filter_assertions = {
        'concepts': set(),
        'roles': set(),
    }

    for role_assertion in file:
        values = role_assertion.split()
        role_name = concept_role_translations[values[1]]
        filter_assertions['roles'].add((values[0], Atomic_Concept(role_name), values[2]))

        for sup_role_name in extended_role_inclusions[role_name]['sup_role']:
            filter_assertions['roles'].add((values[0], Atomic_Role(sup_role_name), values[2]))

        for sup_inv_role_name in extended_role_inclusions[role_name]['sup_inv_role']:
            filter_assertions['roles'].add((values[2], Atomic_Role(sup_inv_role_name), values[0]))

    filter_assertions['concepts'] = sorted(list(filter_assertions['concepts']),
                                           key=lambda x: x[1].get_op_name() + x[0], )
    filter_assertions['roles'] = sorted(list(filter_assertions['roles']),
                                        key=lambda x: x[1].get_op_name() + x[0] + x[2])
    return filter_assertions