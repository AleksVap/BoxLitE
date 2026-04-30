from DL_Lite_Model.Atomic_Concept import Atomic_Concept
from DL_Lite_Model.Atomic_Role import Atomic_Role
from DL_Lite_Model.Existential_Concept import Existential_Concept
from DL_Lite_Model.Inverse_Role import Inverse_Role
from LoaderUtils import loadFilterAssertions, loadDataFromFile, loadIndividuals
from load_relation_dictionaries import get_role_concept_relations


def loadData(dataset, evaluation_mode='Train'):
    data = {'TBox': {'concepts': [],
                     'roles': [
                         (Atomic_Role('spouse'), Atomic_Role('relative')),
                         (Atomic_Role('hasSibling'), Atomic_Role('relative')),
                         (Atomic_Role('hasChild'), Atomic_Role('relative')),
                         (Atomic_Role('hasParent'), Atomic_Role('relative')),
                         (Atomic_Role('hasFather'), Atomic_Role('hasParent')),
                         (Atomic_Role('hasMother'), Atomic_Role('hasParent')),
                         (Atomic_Role('relative'), Inverse_Role(Atomic_Role('relative'))),
                         (Atomic_Role('hasSibling'), Inverse_Role(Atomic_Role('hasSibling'))),
                         (Atomic_Role('spouse'), Inverse_Role(Atomic_Role('spouse'))),
                         (Atomic_Role('hasChild'), Inverse_Role(Atomic_Role('hasParent'))),
                     ]
                     },
            'ABox': {},
            'Filter_Assertions': loadFilterAssertions(dataset),
            'Triples_To_Evaluate': {'concepts': [], 'roles': []},
            }

    concepts = []
    roles = ['hasSibling', 'hasChild', 'hasMother', 'hasFather', 'hasParent', 'spouse', 'relative']
    individuals = loadIndividuals(dataset)

    data['ABox'] = loadDataFromFile(dataset, 'train')
    data['Val'] = loadDataFromFile(dataset, 'validation')

    # Evaluates if all the learned trained data is actually learned
    if evaluation_mode == 'Train':
        for predicate_type in ['concepts', 'roles']:
            data['Filter_Assertions'][predicate_type] = []
        data['Triples_To_Evaluate'] = data['ABox']
    # Evaluates if all the test triples are learned
    elif evaluation_mode == 'Test':
        for predicate_type in ['concepts', 'roles']:
            data['Filter_Assertions'][predicate_type] = data['Val'][predicate_type] + data['ABox'][predicate_type]
        data['Triples_To_Evaluate'] = loadDataFromFile(dataset, 'test')
    # Evaluates if all the validation assertions are learned
    elif evaluation_mode == 'Validation':
        for predicate_type in ['concepts', 'roles']:
            data['Filter_Assertions'][predicate_type] = data['ABox'][predicate_type]
            data['Triples_To_Evaluate'][predicate_type] = data['Val'][predicate_type]

    nce = [Atomic_Concept(concept) for concept in concepts] + \
          [Existential_Concept(Atomic_Role(role)) for role in roles] + \
          [Existential_Concept(Inverse_Role(Atomic_Role(role))) for role in roles]

    nrn = [Atomic_Role(role) for role in roles] + [Inverse_Role(Atomic_Role(role)) for role in roles]

    data['TBox']['concepts'] = data['TBox']['concepts'] + [(C, C) for C in nce]
    data['TBox']['roles'] = data['TBox']['roles'] + [(r, r) for r in nrn]

    # Remove duplicates
    data['TBox']['roles'] = list(dict.fromkeys(data['TBox']['roles']))
    data['TBox']['concepts'] = list(dict.fromkeys(data['TBox']['concepts']))

    concept_relations, role_relations = get_role_concept_relations(concepts, roles, data)

    # Box consistency requires one dimension per concept box and per Head/Tail-Box of a role
    concept_paths = [[Atomic_Concept(concept)] for concept in concepts] + \
                    [[Existential_Concept(Atomic_Role(role))] for role in roles] + \
                    [[Existential_Concept(Inverse_Role(Atomic_Role(role)))] for role in roles]

    return data, concepts, roles, individuals, concept_paths, concept_relations, role_relations
