from DL_Lite_Model.Atomic_Concept import Atomic_Concept
from DL_Lite_Model.Atomic_Role import Atomic_Role
from DL_Lite_Model.Existential_Concept import Existential_Concept
from DL_Lite_Model.Inverse_Role import Inverse_Role
from DL_Lite_Model.Neg_Concept import Neg_Concept


def get_role_concept_relations(concepts, roles, data):
    ##################################################
    # CONCEPTS
    ##################################################

    concept_relations = {}
    all_concepts = set()
    for role in roles:
        concept_relations[Existential_Concept(Atomic_Role(role))] = {'subset': {Existential_Concept(Atomic_Role(role))},
                                                                     'superset': {
                                                                         Existential_Concept(Atomic_Role(role))},
                                                                     'no_relation': set()}
        concept_relations[Existential_Concept(Inverse_Role(Atomic_Role(role)))] = {
            'subset': {Existential_Concept(Inverse_Role(Atomic_Role(role)))},
            'superset': {Existential_Concept(Inverse_Role(Atomic_Role(role)))},
            'no_relation': set()}
        all_concepts = all_concepts.union(
            {Existential_Concept(Atomic_Role(role)), Existential_Concept(Inverse_Role(Atomic_Role(role)))})

    for concept in concepts:
        concept_relations[Atomic_Concept(concept)] = {'subset': {Atomic_Concept(concept)},
                                                      'superset': {Atomic_Concept(concept)}, 'no_relation': set()}
        all_concepts = all_concepts.union({Atomic_Concept(concept)})

    for axiom in data['TBox']['concepts']:
        sub_concept = axiom[0]
        sup_concept = axiom[1]

        if not isinstance(sup_concept, Neg_Concept):
            concept_relations[sup_concept]['subset'] = concept_relations[sup_concept]['subset'].union({sub_concept})
            concept_relations[sub_concept]['superset'] = concept_relations[sub_concept]['superset'].union({sup_concept})

    for concept in concept_relations.keys():
        concept_relations[concept]['no_relation'] = all_concepts.difference(
            concept_relations[concept]['subset'].union(concept_relations[concept]['superset']))

    for concept in concept_relations.keys():
        concept_relations[concept]['no_relation'] = sorted(list(concept_relations[concept]['no_relation']),
                                                           key=lambda x: x.get_op_name())
        concept_relations[concept]['subset'] = sorted(list(concept_relations[concept]['subset']),
                                                      key=lambda x: x.get_op_name())
        concept_relations[concept]['superset'] = sorted(list(concept_relations[concept]['superset']),
                                                        key=lambda x: x.get_op_name())

    ##################################################
    # ROLES
    ##################################################

    role_relations = {}
    all_roles = set()
    for role in roles:
        role_relations[Atomic_Role(role)] = {'subset': {Atomic_Role(role)}, 'superset': {Atomic_Role(role)},
                                             'no_relation': set()}
        all_roles = all_roles.union({Atomic_Role(role)})

    for axiom in data['TBox']['roles']:
        sub_role = axiom[0]
        sup_role = axiom[1]

        if isinstance(sub_role, Inverse_Role):
            sub_role = Atomic_Role(sub_role.get_name()[0])
            role_relations[sub_role]['superset'] = role_relations[sub_role]['superset'].union({Inverse_Role(sup_role)})
        else:
            role_relations[sub_role]['superset'] = role_relations[sub_role]['superset'].union({sup_role})

        if isinstance(sup_role, Inverse_Role):
            sup_role = Atomic_Role(sup_role.get_name()[0])
            role_relations[sup_role]['subset'] = role_relations[sup_role]['subset'].union({Inverse_Role(sub_role)})
        else:
            role_relations[sup_role]['subset'] = role_relations[sup_role]['subset'].union({sub_role})

    for role in role_relations.keys():
        role_relations[role]['no_relation'] = all_roles.difference(
            role_relations[role]['subset'].union(role_relations[role]['superset']))

    for role in role_relations.keys():
        role_relations[role]['no_relation'] = sorted(list(role_relations[role]['no_relation']),
                                                     key=lambda x: x.get_op_name())
        role_relations[role]['subset'] = sorted(list(role_relations[role]['subset']), key=lambda x: x.get_op_name())
        role_relations[role]['superset'] = sorted(list(role_relations[role]['superset']), key=lambda x: x.get_op_name())

    return concept_relations, role_relations
