import copy
import os
import sys
import json
import matplotlib
import numpy as np
import cvxpy as cp
from cvxpy import mixed_norm
from scipy.sparse import csc_array
from tqdm import tqdm
import FamilyDataLoader
import Utils
from Utils import S_OMEGA, epsilon
from DL_Lite_Model.Neg_Concept import Neg_Concept
from DL_Lite_Model.Atomic_Role import Atomic_Role
from DL_Lite_Model.Inverse_Role import Inverse_Role
from DL_Lite_Model.Atomic_Concept import Atomic_Concept
from DL_Lite_Model.Existential_Concept import Existential_Concept
from Plotting_Utils import plot_and_save_solution_weak, getTransformedBox

x_variables_dict = {}
sorted_parameter_names = sorted(["dimensionality", "lambda_1", "lambda_2", "lambda_3"])
matplotlib.use('TkAgg')

def loadDataset(dataset, evaluation_mode='Train'):
    if dataset[0:3] == 'F_v':
        data, concepts, roles, individuals, concept_paths, concept_relations, role_relations = FamilyDataLoader.loadData(
            dataset=dataset,
            evaluation_mode=evaluation_mode)
    else:
        raise Exception('Unknown Dataset ' + dataset)

    return data, concepts, roles, individuals, concept_paths, concept_relations, role_relations


def getIndividual(individual_name, x_index_dict, dimensionality):
    individual_index = x_index_dict["individuals"][individual_name]
    individual_index_position = individual_index + x_index_dict["individuals_pos_relative_offset"]
    individual_index_bump = individual_index + x_index_dict["individuals_bump_relative_offset"]
    cols_position = list(range(individual_index_position, individual_index_position + dimensionality))
    cols_np_position = np.array(cols_position)
    cols_bump = list(range(individual_index_bump, individual_index_bump + dimensionality))
    cols_np_bumps = np.array(cols_bump)
    return {
        "col_p": cols_np_position,
        "data_p": np.ones(dimensionality),
        "col_b": cols_np_bumps,
        "data_b": np.ones(dimensionality)
    }


def construct_param_vector(individuals, concepts, roles, dimensionality):
    x_arr_dim = ((len(individuals) * 2 + len(concepts) * 2 + len(roles) * 6) * dimensionality) + 1
    x_arr = cp.Variable(shape=x_arr_dim)
    individuals_pos_offset = 0
    individuals_bump_offset = len(individuals) * dimensionality
    concepts_lb_offset = individuals_bump_offset + len(individuals) * dimensionality
    concepts_ub_offset = concepts_lb_offset + len(concepts) * dimensionality
    roles_head_lb_offset = concepts_ub_offset + len(concepts) * dimensionality
    roles_head_ub_offset = roles_head_lb_offset + len(roles) * dimensionality
    roles_tail_lb_offset = roles_head_ub_offset + len(roles) * dimensionality
    roles_tail_ub_offset = roles_tail_lb_offset + len(roles) * dimensionality
    roles_bump_lb_offset = roles_tail_ub_offset + len(roles) * dimensionality
    roles_bump_ub_offset = roles_bump_lb_offset + len(roles) * dimensionality

    individuals_pos_relative_offset = 0
    individuals_bump_relative_offset = len(individuals) * dimensionality
    concepts_lb_relative_offset = 0
    concepts_ub_relative_offset = len(concepts) * dimensionality
    roles_head_lb_relative_offset = 0
    roles_head_ub_relative_offset = roles_head_lb_relative_offset + len(roles) * dimensionality
    roles_tail_lb_relative_offset = roles_head_ub_relative_offset + len(roles) * dimensionality
    roles_tail_ub_relative_offset = roles_tail_lb_relative_offset + len(roles) * dimensionality
    roles_bump_lb_relative_offset = roles_tail_ub_relative_offset + len(roles) * dimensionality
    roles_bump_ub_relative_offset = roles_bump_lb_relative_offset + len(roles) * dimensionality

    x_index_dict = {"individuals": {},
                    "concepts": {},
                    "roles": {},
                    "individuals_pos_offset": individuals_pos_offset,
                    "individuals_bump_offset": individuals_bump_offset,
                    "concepts_lb_offset": concepts_lb_offset,
                    "concepts_ub_offset": concepts_ub_offset,
                    "roles_head_lb_offset": roles_head_lb_offset,
                    "roles_head_ub_offset": roles_head_ub_offset,
                    "roles_tail_lb_offset": roles_tail_lb_offset,
                    "roles_tail_ub_offset": roles_tail_ub_offset,
                    "roles_bump_lb_offset": roles_bump_lb_offset,
                    "roles_bump_ub_offset": roles_bump_ub_offset,

                    "individuals_pos_relative_offset": individuals_pos_relative_offset,
                    "individuals_bump_relative_offset": individuals_bump_relative_offset,
                    "concepts_lb_relative_offset": concepts_lb_relative_offset,
                    "concepts_ub_relative_offset": concepts_ub_relative_offset,
                    "roles_head_lb_relative_offset": roles_head_lb_relative_offset,
                    "roles_head_ub_relative_offset": roles_head_ub_relative_offset,
                    "roles_tail_lb_relative_offset": roles_tail_lb_relative_offset,
                    "roles_tail_ub_relative_offset": roles_tail_ub_relative_offset,
                    "roles_bump_lb_relative_offset": roles_bump_lb_relative_offset,
                    "roles_bump_ub_relative_offset": roles_bump_ub_relative_offset,

                    "constant_index": x_arr_dim - 1,
                    }

    for i, individual in enumerate(tqdm(individuals)):
        x_index_dict["individuals"][individual] = i * dimensionality

    for i, concept in enumerate(tqdm(concepts)):
        x_index_dict["concepts"][concept] = concepts_lb_offset + (i * dimensionality)

    for i, role in enumerate(tqdm(roles)):
        x_index_dict["roles"][role] = roles_head_lb_offset + (i * dimensionality)

    return x_arr, x_index_dict


def concept_assertion_computation_existential(c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                              A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists):
    mid_index = dimensionality

    for index in range(dimensionality):
        A_col_l = np.array([c_paras['col_l'][index], c_paras['col_l'][mid_index + index], i_paras['col_p'][index],
                            c_paras['col_l'][-1]])
        A_data_l = np.array([c_paras['data_l'][index], c_paras['data_l'][mid_index + index], -i_paras['data_p'][index],
                             c_paras['data_l'][-1]])

        A_col_l_lists.append(A_col_l)
        A_data_l_lists.append(A_data_l)

        A_col_r = np.array([c_paras['col_u'][index], c_paras['col_u'][mid_index + index], i_paras['col_p'][index],
                            c_paras['col_u'][-1]])
        A_data_r = np.array([-c_paras['data_u'][index], -c_paras['data_u'][mid_index + index], i_paras['data_p'][index],
                             c_paras['data_u'][-1]])

        A_col_r_lists.append(A_col_r)
        A_data_r_lists.append(A_data_r)

        row = np.ones(len(A_col_l)) * (row_value * dimensionality + index)

        A_row_lists.append(row)


def concept_assertion_computation_non_existential(c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                                  A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists):
    for index in range(dimensionality):
        A_col_l = np.array([c_paras['col_l'][index], i_paras['col_p'][index], c_paras['col_l'][-1]])
        A_data_l = np.array([c_paras['data_l'][index], -i_paras['data_p'][index], c_paras['data_l'][-1]])

        A_col_l_lists.append(A_col_l)
        A_data_l_lists.append(A_data_l)

        A_col_r = np.array([c_paras['col_u'][index], i_paras['col_p'][index], c_paras['col_u'][-1]])
        A_data_r = np.array([-c_paras['data_u'][index], i_paras['data_p'][index], c_paras['data_u'][-1]])

        A_col_r_lists.append(A_col_r)
        A_data_r_lists.append(A_data_r)

        row = np.ones(len(A_col_l)) * (row_value * dimensionality + index)

        A_row_lists.append(row)


def concept_assertion_term_computation(concept, c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                       A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists
                                       ):
    existential_flag = False
    concept_copy = copy.deepcopy(concept)
    while True:
        if isinstance(concept_copy, Existential_Concept):
            existential_flag = True
            break

        if hasattr(concept_copy, 'concept'):
            concept_copy = concept_copy.concept
        else:
            break

    if existential_flag:
        concept_assertion_computation_existential(c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                                  A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists)

    else:
        concept_assertion_computation_non_existential(c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                                      A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists)


def concept_assertion_term(concept_assertions, dimensionality, x_arr, x_index_dict):
    A_col_l_lists = []
    A_data_l_lists = []

    A_col_r_lists = []
    A_data_r_lists = []

    A_row_lists = []

    for row_value, ca in enumerate(concept_assertions):
        c_paras = getTransformedBox(ca[1], dimensionality, x_arr, x_index_dict)
        i_paras = getIndividual(ca[0], x_index_dict, dimensionality)

        concept_assertion_term_computation(ca[1], c_paras, i_paras, dimensionality, row_value, A_col_l_lists,
                                           A_data_l_lists, A_col_r_lists, A_data_r_lists, A_row_lists
                                           )
    if A_col_l_lists != []:
        A_col_l_lists = np.hstack(A_col_l_lists).astype(np.int64)
        A_data_l_lists = np.hstack(A_data_l_lists)
        A_col_r_lists = np.hstack(A_col_r_lists).astype(np.int64)
        A_data_r_lists = np.hstack(A_data_r_lists)
        A_row_lists = np.hstack(A_row_lists).astype(np.int64)

        A_l = csc_array((A_data_l_lists, (A_row_lists, A_col_l_lists)))
        A_r = csc_array((A_data_r_lists, (A_row_lists, A_col_r_lists)))
    else:
        num_rows = x_arr.shape[0] * dimensionality
        num_cols = x_arr.shape[0]

        A_l = csc_array(([], ([], [])), shape=(num_rows, num_cols))
        A_r = csc_array(([], ([], [])), shape=(num_rows, num_cols))
    return A_l, A_r


def role_assertion_term(role_assertions, dimensionality, x_arr, x_index_dict):
    # Head and Tail
    A_col_l_dict = {}
    A_data_l_dict = {}
    A_col_r_dict = {}
    A_data_r_dict = {}
    A_row_dict = {}

    for mode in ['h', 't']:
        A_col_l_lists = []
        A_data_l_lists = []

        A_col_r_lists = []
        A_data_r_lists = []

        A_row_lists = []

        individual = 0 if mode == 'h' else 2
        individual_prime = 2 if mode == 'h' else 0

        for row_value, ra in enumerate(role_assertions):
            r_paras = getTransformedBox(ra[1], dimensionality, x_arr, x_index_dict)
            individual_paras = getIndividual(ra[individual], x_index_dict, dimensionality)
            individual_prime_paras = getIndividual(ra[individual_prime], x_index_dict, dimensionality)

            role_col_l = 'col_' + mode + "l"
            role_col_u = 'col_' + mode + "u"
            role_data_l = 'data_' + mode + "l"
            role_data_u = 'data_' + mode + "u"

            for index in range(dimensionality):
                A_col_l = np.array([r_paras[role_col_l][index], individual_paras['col_p'][index],
                                    individual_prime_paras['col_b'][index], r_paras[role_col_l][-1]])
                A_data_l = np.array([r_paras[role_data_l][index], -individual_paras['data_p'][index],
                                     -individual_prime_paras['data_b'][index], r_paras[role_data_l][-1]])

                A_col_l_lists.append(A_col_l)
                A_data_l_lists.append(A_data_l)

                A_col_r = np.array([r_paras[role_col_u][index], individual_paras['col_p'][index],
                                    individual_prime_paras['col_b'][index], r_paras[role_col_u][-1]])
                A_data_r = np.array([-r_paras[role_data_u][index], individual_paras['data_p'][index],
                                     individual_prime_paras['data_b'][index], r_paras[role_data_u][-1]])

                A_col_r_lists.append(A_col_r)
                A_data_r_lists.append(A_data_r)

                row = np.ones(len(A_col_l)) * (row_value * dimensionality + index)

                A_row_lists.append(row)

        A_col_l_dict[mode] = np.array(A_col_l_lists, dtype=np.int64).flatten()
        A_data_l_dict[mode] = np.array(A_data_l_lists).flatten()
        A_col_r_dict[mode] = np.array(A_col_r_lists, dtype=np.int64).flatten()
        A_data_r_dict[mode] = np.array(A_data_r_lists).flatten()
        A_row_dict[mode] = np.array(A_row_lists, dtype=np.int64).flatten()

    A_l_h = csc_array((A_data_l_dict['h'], (A_row_dict['h'], A_col_l_dict['h'])))
    A_r_h = csc_array((A_data_r_dict['h'], (A_row_dict['h'], A_col_r_dict['h'])))
    A_l_t = csc_array((A_data_l_dict['t'], (A_row_dict['t'], A_col_l_dict['t'])))
    A_r_t = csc_array((A_data_r_dict['t'], (A_row_dict['t'], A_col_r_dict['t'])))

    # Bumps
    A_col_l_dict = {}
    A_data_l_dict = {}
    A_col_r_dict = {}
    A_data_r_dict = {}
    A_row_dict = {}

    for mode in ['h', 't']:
        A_col_l_lists = []
        A_data_l_lists = []

        A_col_r_lists = []
        A_data_r_lists = []

        A_row_lists = []

        individual = 0 if mode == 'h' else 2

        for row_value, ra in enumerate(role_assertions):
            r_paras = getTransformedBox(ra[1], dimensionality, x_arr, x_index_dict)
            i_paras = getIndividual(ra[individual], x_index_dict, dimensionality)

            for index in range(dimensionality):
                A_col_l = np.array([r_paras['col_bl'][index], i_paras['col_b'][index], r_paras['col_bl'][-1]])
                A_data_l = np.array([r_paras['data_bl'][index], -i_paras['data_b'][index], r_paras['data_bl'][-1]])

                A_col_l_lists.append(A_col_l)
                A_data_l_lists.append(A_data_l)

                A_col_r = np.array([r_paras['col_bu'][index], i_paras['col_b'][index], r_paras['col_bu'][-1]])
                A_data_r = np.array([-r_paras['data_bu'][index], i_paras['data_b'][index], r_paras['data_bu'][-1]])

                A_col_r_lists.append(A_col_r)
                A_data_r_lists.append(A_data_r)

                row = np.ones(len(A_col_l)) * (row_value * dimensionality + index)

                A_row_lists.append(row)

        A_col_l_dict[mode] = np.array(A_col_l_lists, dtype=np.int64).flatten()
        A_data_l_dict[mode] = np.array(A_data_l_lists).flatten()
        A_col_r_dict[mode] = np.array(A_col_r_lists, dtype=np.int64).flatten()
        A_data_r_dict[mode] = np.array(A_data_r_lists).flatten()
        A_row_dict[mode] = np.array(A_row_lists, dtype=np.int64).flatten()

    A_l_b1 = csc_array((A_data_l_dict['h'], (A_row_dict['h'], A_col_l_dict['h'])))
    A_r_b1 = csc_array((A_data_r_dict['h'], (A_row_dict['h'], A_col_r_dict['h'])))
    A_l_b2 = csc_array((A_data_l_dict['t'], (A_row_dict['t'], A_col_l_dict['t'])))
    A_r_b2 = csc_array((A_data_r_dict['t'], (A_row_dict['t'], A_col_r_dict['t'])))

    return A_l_h, A_r_h, A_l_t, A_r_t, A_l_b1, A_r_b1, A_l_b2, A_r_b2


def concept_inclusion_constraints(tbox_concepts, dimensionality, x_arr, x_index_dict):
    """
    l1 - l0 <= 0
    u0 - u1 <= 0

    Returns A such that A @ x <= 0 encodes all constraints.
    """
    if not tbox_concepts:
        return None

    A_col_lists = []
    A_data_lists = []
    A_row_lists = []

    row_counter = 0

    for axiom in tbox_concepts:
        c0_paras = getTransformedBox(axiom[0], dimensionality, x_arr, x_index_dict)
        c1_paras = getTransformedBox(axiom[1], dimensionality, x_arr, x_index_dict)

        existential_flag_0 = False
        existential_flag_1 = False
        concept_copy = copy.deepcopy(axiom[0])
        while True:
            if isinstance(concept_copy, Existential_Concept):
                existential_flag_0 = True
                break
            if hasattr(concept_copy, 'concept'):
                concept_copy = concept_copy.concept
            else:
                break

        concept_copy = copy.deepcopy(axiom[1])
        while True:
            if isinstance(concept_copy, Existential_Concept):
                existential_flag_1 = True
                break
            if hasattr(concept_copy, 'concept'):
                concept_copy = concept_copy.concept
            else:
                break

        # 2 constraints per dimension: l and u
        for index in range(dimensionality):
            col_l_existential_amendment = []
            data_l_existential_amendment = []
            col_u_existential_amendment = []
            data_u_existential_amendment = []
            if existential_flag_0:
                col_l_existential_amendment += [c0_paras['col_l'][dimensionality + index]]
                data_l_existential_amendment += [-c0_paras['data_l'][dimensionality + index]]
                col_u_existential_amendment += [c0_paras['col_u'][dimensionality + index]]
                data_u_existential_amendment += [c0_paras['data_u'][dimensionality + index]]
            if existential_flag_1:
                col_l_existential_amendment += [c1_paras['col_l'][dimensionality + index]]
                data_l_existential_amendment += [c1_paras['data_l'][dimensionality + index]]
                col_u_existential_amendment += [c1_paras['col_u'][dimensionality + index]]
                data_u_existential_amendment += [-c1_paras['data_u'][dimensionality + index]]

            # l1 - l0 <= 0
            A_col_l = np.array([c1_paras['col_l'][index], c0_paras['col_l'][index]] + col_l_existential_amendment)
            A_data_l = np.array([c1_paras['data_l'][index], -c0_paras['data_l'][index]] + data_l_existential_amendment)

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # u0 - u1 <= 0
            A_col_u = np.array([c0_paras['col_u'][index], c1_paras['col_u'][index]] + col_u_existential_amendment)
            A_data_u = np.array([c0_paras['data_u'][index], -c1_paras['data_u'][index]] + data_u_existential_amendment)

            A_col_lists.append(A_col_u)
            A_data_lists.append(A_data_u)
            A_row_lists.append(np.ones(len(A_col_u)) * row_counter)
            row_counter += 1

    A_col_lists = np.hstack(A_col_lists).astype(np.int64)
    A_data_lists = np.hstack(A_data_lists).astype(np.int64)
    A_row_lists = np.hstack(A_row_lists).astype(np.int64)

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))

    return A


def role_inclusion_constraints(tbox_concepts, dimensionality, x_arr, x_index_dict):
    """
    h1_l - h0_l <= 0
    h0_u - h1_u <= 0
    t1_l - t0_l <= 0
    t0_u - t1_u <= 0
    b1_l - b0_l <= 0
    b0_u - b1_u <= 0

    Returns A such that A @ x <= 0 encodes all constraints.
    """
    if not tbox_concepts:
        return None

    A_col_lists = []
    A_data_lists = []
    A_row_lists = []

    row_counter = 0

    for axiom in tbox_concepts:
        r0_paras = getTransformedBox(axiom[0], dimensionality, x_arr, x_index_dict)
        r1_paras = getTransformedBox(axiom[1], dimensionality, x_arr, x_index_dict)
        # 2 constraints per dimension: l and u
        for index in range(dimensionality):
            # h1_l - h0_l <= 0
            A_col_l = np.array([r1_paras['col_hl'][index], r0_paras['col_hl'][index]])
            A_data_l = np.array([r1_paras['data_hl'][index], -r0_paras['data_hl'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # h0_u - h1_u <= 0
            A_col_l = np.array([r0_paras['col_hu'][index], r1_paras['col_hu'][index]])
            A_data_l = np.array([r0_paras['data_hu'][index], -r1_paras['data_hu'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # t1_l - t0_l <= 0
            A_col_l = np.array([r1_paras['col_tl'][index], r0_paras['col_tl'][index]])
            A_data_l = np.array([r1_paras['data_tl'][index], -r0_paras['data_tl'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # t0_u - t1_u <= 0
            A_col_l = np.array([r0_paras['col_tu'][index], r1_paras['col_tu'][index]])
            A_data_l = np.array([r0_paras['data_tu'][index], -r1_paras['data_tu'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # b1_l - b0_l <= 0
            A_col_l = np.array([r1_paras['col_bl'][index], r0_paras['col_bl'][index]])
            A_data_l = np.array([r1_paras['data_bl'][index], -r0_paras['data_bl'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

            # b0_u - b1_u <= 0
            A_col_l = np.array([r0_paras['col_bu'][index], r1_paras['col_bu'][index]])
            A_data_l = np.array([r0_paras['data_bu'][index], -r1_paras['data_bu'][index]])

            A_col_lists.append(A_col_l)
            A_data_lists.append(A_data_l)
            A_row_lists.append(np.ones(len(A_col_l)) * row_counter)
            row_counter += 1

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))

    return A


def box_consistency_constraints(concept_paths, dimensionality, x_arr, x_index_dict):
    """
    center[dim] <= -S_OMEGA / 2 - epsilon
    (l_c[dim] + u_c[dim]) / 2 <= -S_OMEGA / 2 - epsilon
    l_c[dim] + u_c[dim] <= -S_OMEGA - 2*epsilon

    Returns A, b such that A @ x <= b encodes all constraints.
    """
    A_col_lists = []
    A_data_lists = []
    A_row_lists = []
    b_list = []

    row_counter = 0

    for index, path in enumerate(concept_paths):
        for complex_concept in path:
            c_paras = getTransformedBox(complex_concept, dimensionality, x_arr, x_index_dict)
            existential_flag = False
            concept_copy = copy.deepcopy(complex_concept)
            while True:
                if isinstance(concept_copy, Existential_Concept):
                    existential_flag = True
                    break
                if hasattr(concept_copy, 'concept'):
                    concept_copy = concept_copy.concept
                else:
                    break

            # l_c[dim] + u_c[dim] <= -S_OMEGA - 2*epsilon
            col_existential_amendment = []
            data_existential_amendment = []
            if existential_flag:
                col_existential_amendment += [c_paras['col_l'][dimensionality + index],
                                              c_paras['col_u'][dimensionality + index]]
                data_existential_amendment += [c_paras['data_l'][dimensionality + index],
                                               c_paras['data_u'][dimensionality + index]]

            A_col = np.array([
                                 c_paras['col_l'][index],
                                 c_paras['col_u'][index]
                             ] + col_existential_amendment)

            A_data = np.array([
                                  c_paras['data_l'][index],
                                  c_paras['data_u'][index]
                              ] + data_existential_amendment)

            A_col_lists.append(A_col)
            A_data_lists.append(A_data)
            A_row_lists.append(np.ones(len(A_col)) * row_counter)

            b_list.append(- S_OMEGA - 2 * epsilon)

            row_counter += 1

    if row_counter == 0:
        return None, None

    A_col_lists = np.hstack(A_col_lists).astype(np.int64)
    A_data_lists = np.hstack(A_data_lists).astype(np.int64)
    A_row_lists = np.hstack(A_row_lists).astype(np.int64)

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))
    b = np.array(b_list)

    return A, b


def role_box_width_constraints(roles, dimensionality, x_arr, x_index_dict):
    """
    hl - hu <= 0
    tl - tu <= 0
    bl - bu <= 0
    hu - hl <= 2 * S_OMEGA
    tu - tl <= 2 * S_OMEGA
    bu - bl <= 2 * S_OMEGA

    Returns A, b such that A @ x <= b encodes all constraints.
    """
    if not roles:
        return None, None

    A_col_lists = []
    A_data_lists = []
    A_row_lists = []
    b_list = []

    row_counter = 0

    for role in roles:
        r_paras = getTransformedBox(Atomic_Role(role), dimensionality, x_arr, x_index_dict)

        box_pairs = [
            ('col_hl', 'col_hu', 'data_hl', 'data_hu'),  # head
            ('col_tl', 'col_tu', 'data_tl', 'data_tu'),  # tail
            ('col_bl', 'col_bu', 'data_bl', 'data_bu'),  # bump
        ]

        for col_l, col_u, data_l, data_u in box_pairs:
            for index in range(dimensionality):
                # Constraint 1: l - u <= 0 (i.e., u - l >= 0)
                A_col = np.array([r_paras[col_l][index], r_paras[col_u][index]])
                A_data = np.array([r_paras[data_l][index], -r_paras[data_u][index]])

                A_col_lists.append(A_col)
                A_data_lists.append(A_data)
                A_row_lists.append(np.ones(len(A_col)) * row_counter)
                b_list.append(0)
                row_counter += 1

                # Constraint 2: u - l <= 2 * S_OMEGA
                A_col = np.array([r_paras[col_u][index], r_paras[col_l][index]])
                A_data = np.array([r_paras[data_u][index], -r_paras[data_l][index]])

                A_col_lists.append(A_col)
                A_data_lists.append(A_data)
                A_row_lists.append(np.ones(len(A_col)) * row_counter)
                b_list.append(2 * S_OMEGA)
                row_counter += 1

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))
    b = np.array(b_list)

    return A, b


def concept_box_width_constraints(concepts, dimensionality, x_arr, x_index_dict):
    """
    Vectorized concept box width constraints:
        l - u <= 0
        u - l <= 2 * S_OMEGA
    """
    if not concepts:
        return None, None

    A_col_lists = []
    A_data_lists = []
    A_row_lists = []
    b_list = []

    row_counter = 0

    for concept in concepts:
        c_paras = getTransformedBox(Atomic_Concept(concept), dimensionality, x_arr, x_index_dict)

        for index in range(dimensionality):
            # l - u <= 0
            A_col = np.array([c_paras['col_l'][index], c_paras['col_u'][index]])
            A_data = np.array([c_paras['data_l'][index], -c_paras['data_u'][index]])

            A_col_lists.append(A_col)
            A_data_lists.append(A_data)
            A_row_lists.append(np.ones(2) * row_counter)
            b_list.append(0)
            row_counter += 1

            # u - l <= 2 * S_OMEGA
            A_col = np.array([c_paras['col_u'][index], c_paras['col_l'][index]])
            A_data = np.array([c_paras['data_u'][index], -c_paras['data_l'][index]])

            A_col_lists.append(A_col)
            A_data_lists.append(A_data)
            A_row_lists.append(np.ones(2) * row_counter)
            b_list.append(2 * S_OMEGA)
            row_counter += 1

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))
    b = np.array(b_list)

    return A, b


def individual_bounds_constraints(individuals, dimensionality, x_arr, x_index_dict):
    """
    pos <= S_OMEGA
    -pos <= S_OMEGA  (i.e., pos >= -S_OMEGA)
    bump <= S_OMEGA
    -bump <= S_OMEGA  (i.e., bump >= -S_OMEGA)
    """
    if not individuals:
        return None, None

    A_col_lists = []
    A_data_lists = []
    A_row_lists = []
    b_list = []

    row_counter = 0

    for individual in individuals:
        i_paras = getIndividual(individual, x_index_dict, dimensionality)

        for index in range(dimensionality):
            # pos <= S_OMEGA
            A_col_lists.append(np.array([i_paras['col_p'][index]]))
            A_data_lists.append(np.array([i_paras['data_p'][index]]))
            A_row_lists.append(np.array([row_counter]))
            b_list.append(S_OMEGA)
            row_counter += 1

            # -pos <= S_OMEGA
            A_col_lists.append(np.array([i_paras['col_p'][index]]))
            A_data_lists.append(np.array([-i_paras['data_p'][index]]))
            A_row_lists.append(np.array([row_counter]))
            b_list.append(S_OMEGA)
            row_counter += 1

            # bump <= S_OMEGA
            A_col_lists.append(np.array([i_paras['col_b'][index]]))
            A_data_lists.append(np.array([i_paras['data_b'][index]]))
            A_row_lists.append(np.array([row_counter]))
            b_list.append(S_OMEGA)
            row_counter += 1

            # -bump <= S_OMEGA
            A_col_lists.append(np.array([i_paras['col_b'][index]]))
            A_data_lists.append(np.array([-i_paras['data_b'][index]]))
            A_row_lists.append(np.array([row_counter]))
            b_list.append(S_OMEGA)
            row_counter += 1

    A_col = np.array(A_col_lists, dtype=np.int64).flatten()
    A_data = np.array(A_data_lists).flatten()
    A_row = np.array(A_row_lists, dtype=np.int64).flatten()

    num_rows = row_counter
    num_cols = x_arr.shape[0]

    A = csc_array((A_data, (A_row, A_col)), shape=(num_rows, num_cols))
    b = np.array(b_list)

    return A, b


def box_width_regularization_term(num_concepts, num_roles, dimensionality, x_arr, x_index_dict):
    box_paras = {
        'c_l': x_arr[
            x_index_dict["concepts_lb_offset"]:(x_index_dict["concepts_lb_offset"] + num_concepts * dimensionality)],
        'c_u': x_arr[
            x_index_dict["concepts_ub_offset"]:(x_index_dict["concepts_ub_offset"] + num_concepts * dimensionality)],
        'h_l': x_arr[
            x_index_dict["roles_head_lb_offset"]:(x_index_dict["roles_head_lb_offset"] + (num_roles * dimensionality))],
        'h_u': x_arr[
            x_index_dict["roles_head_ub_offset"]:(x_index_dict["roles_head_ub_offset"] + (num_roles * dimensionality))],
        't_l': x_arr[
            x_index_dict["roles_tail_lb_offset"]:(x_index_dict["roles_tail_lb_offset"] + (num_roles * dimensionality))],
        't_u': x_arr[
            x_index_dict["roles_tail_ub_offset"]:(x_index_dict["roles_tail_ub_offset"] + (num_roles * dimensionality))],
        'b_l': x_arr[
            x_index_dict["roles_bump_lb_offset"]:(x_index_dict["roles_bump_lb_offset"] + (num_roles * dimensionality))],
        'b_u': x_arr[
            x_index_dict["roles_bump_ub_offset"]:(x_index_dict["roles_bump_ub_offset"] + (num_roles * dimensionality))],
    }

    if num_concepts == 0:
        box_paras['c_l'] = cp.vec(np.zeros(dimensionality))
        box_paras['c_u'] = cp.vec(np.zeros(dimensionality))

    stacked_box_paras = {
        'L': cp.hstack([box_paras['c_l'], box_paras['h_l'], box_paras['t_l']]),
        'U': cp.hstack([box_paras['c_u'], box_paras['h_u'], box_paras['t_u']])
    }

    stacked_box_paras_bump = {
        'L': cp.hstack([box_paras['b_l']]),
        'U': cp.hstack([box_paras['b_u']])
    }

    for key in stacked_box_paras.keys():
        stacked_box_paras[key] = cp.reshape(stacked_box_paras[key],
                                            shape=(dimensionality,
                                                   int(stacked_box_paras[key].shape[0] / dimensionality)),
                                            order='C')

    for key in stacked_box_paras_bump.keys():
        stacked_box_paras_bump[key] = cp.reshape(stacked_box_paras_bump[key],
                                                 shape=(dimensionality,
                                                        int(stacked_box_paras_bump[key].shape[0] / dimensionality)),
                                                 order='C')

    return (mixed_norm(stacked_box_paras['U'] - stacked_box_paras['L']),
            mixed_norm(stacked_box_paras_bump['U'] - stacked_box_paras_bump['L']))


def create_Problem(dataset, dimensionality, parameters):
    data, concepts, roles, individuals, concept_paths, concept_relations, role_relations = loadDataset(dataset)
    constraints = []
    objective_parts = []

    x_arr, x_index_dict = construct_param_vector(individuals, concepts, roles, dimensionality)

    x_variables_dict["x_arr"] = x_arr
    x_variables_dict["x_index_dict"] = x_index_dict

    # Negative sampling curation
    string_to_role = {}
    existential_concepts = {}

    for h, r, *_ in data["ABox"]['roles']:
        existential_complex_concept = Existential_Concept(r)
        role_name = r.get_op_name()

        if role_name not in existential_concepts:
            string_to_role[role_name] = existential_complex_concept
            existential_concepts[role_name] = []

        existential_concepts[role_name].append(h)

    individuals_set = set(individuals)
    individuals_sorted = sorted(individuals)
    negated_existential_concepts_data = []

    for str_role in sorted(existential_concepts.keys()):
        individuals_list = existential_concepts[str_role]
        existential_concepts[str_role] = set(individuals_list)

        neg_role_obj = Neg_Concept(string_to_role[str_role])

        negated_individuals = sorted(individuals_set - existential_concepts[str_role])
        negated_existential_concepts_data.extend(
            (concept, neg_role_obj) for concept in negated_individuals
        )

    for role in sorted(roles):
        if role in existential_concepts.keys():
            continue
        negated_existential_role = Neg_Concept(Existential_Concept(Atomic_Role(role)))
        negated_existential_inverse_role = Neg_Concept(Existential_Concept(Inverse_Role(Atomic_Role(role))))
        for individual in individuals_sorted:
            negated_existential_concepts_data.append((individual, negated_existential_role))
            negated_existential_concepts_data.append((individual, negated_existential_inverse_role))

    string_to_concepts = {}
    atomic_concepts = {}
    for i, c in data["ABox"]['concepts']:
        concept = c
        concept_name = c.get_op_name()

        if concept_name not in atomic_concepts:
            string_to_concepts[concept_name] = concept
            atomic_concepts[concept_name] = []

        atomic_concepts[concept_name].append(i)

    for str_concept in sorted(atomic_concepts.keys()):
        individuals_list = atomic_concepts[str_concept]
        atomic_concepts[str_concept] = set(individuals_list)

        neg_concept_obj = Neg_Concept(string_to_concepts[str_concept])

        concept_set = atomic_concepts[str_concept]
        negated_individuals = sorted(individuals_set - concept_set)
        negated_existential_concepts_data.extend(
            (concept, neg_concept_obj) for concept in negated_individuals
        )

    for concept in sorted(concepts):
        if concept in atomic_concepts.keys():
            continue
        negated_concept = Neg_Concept(concept)
        for individual in individuals_sorted:
            negated_existential_concepts_data.append((individual, negated_concept))

    negated_concept_assertions_obj_term = []
    if negated_existential_concepts_data != []:
        A_l_concept_sampling, A_r_concept_sampling = concept_assertion_term(negated_existential_concepts_data,
                                                                            dimensionality, x_arr,
                                                                            x_index_dict)

        y_c_1 = cp.Variable(A_l_concept_sampling.shape[0])
        constraints_c_1 = [y_c_1 >= 0, cp.sum(y_c_1) >= 1, cp.norm2(y_c_1) <= 1]
        h_c_1 = cp.transforms.suppfunc(y_c_1, constraints_c_1)

        negated_concept_assertions_obj_term = cp.maximum(h_c_1(A_l_concept_sampling @ x_arr),
                                                         h_c_1(A_r_concept_sampling @ x_arr))

    # Assertion distance for concepts
    if data["ABox"]['concepts'] != []:
        A_l_concept, A_r_concept = concept_assertion_term(data["ABox"]['concepts'], dimensionality, x_arr,
                                                          x_index_dict)

        y_c_2 = cp.Variable(A_l_concept.shape[0])
        constraints_c_2 = [y_c_2 >= 0, cp.sum(y_c_2) >= 1, cp.norm2(y_c_2) <= 1]
        h_c_2 = cp.transforms.suppfunc(y_c_2, constraints_c_2)

        concept_assertions_obj_term = cp.maximum(h_c_2(A_l_concept @ x_arr), h_c_2(A_r_concept @ x_arr))

    else:
        concept_assertions_obj_term = cp.Constant(0)

    # Assertion distance for roles
    if data["ABox"]['roles'] != []:
        A_l_h, A_r_h, A_l_t, A_r_t, A_l_b1, A_r_b1, A_l_b2, A_r_b2 = role_assertion_term(data["ABox"]['roles'],
                                                                                         dimensionality, x_arr,
                                                                                         x_index_dict)

        y_1 = cp.Variable(A_l_h.shape[0])
        constraints_1 = [y_1 >= 0, cp.sum(y_1) >= 1, cp.norm2(y_1) <= 1]
        h_1 = cp.transforms.suppfunc(y_1, constraints_1)

        y_2 = cp.Variable(A_l_t.shape[0])
        constraints_2 = [y_2 >= 0, cp.sum(y_2) >= 1, cp.norm2(y_2) <= 1]
        h_2 = cp.transforms.suppfunc(y_2, constraints_2)

        y_3 = cp.Variable(A_l_b1.shape[0])
        constraints_3 = [y_3 >= 0, cp.sum(y_3) >= 1, cp.norm2(y_3) <= 1]
        h_3 = cp.transforms.suppfunc(y_3, constraints_3)

        y_4 = cp.Variable(A_l_b2.shape[0])
        constraints_4 = [y_4 >= 0, cp.sum(y_4) >= 1, cp.norm2(y_4) <= 1]
        h_4 = cp.transforms.suppfunc(y_4, constraints_4)

        role_assertions_obj_term = cp.maximum(
            h_1(A_l_h @ x_arr), h_1(A_r_h @ x_arr),  # Head Term
            h_2(A_l_t @ x_arr), h_2(A_r_t @ x_arr),  # Tail Term
            h_3(A_l_b1 @ x_arr), h_3(A_r_b1 @ x_arr),  # Box Term 1
            h_4(A_l_b2 @ x_arr), h_4(A_r_b2 @ x_arr))  # Box Term 2

    else:
        role_assertions_obj_term = cp.Constant(0)

    ##### A-Box assertions as maximum of concept and role assertion terms
    concept_role_assertions_obj_term = cp.maximum(concept_assertions_obj_term, role_assertions_obj_term)

    # A-Box-wise aggregation
    assertions_obj_term = cp.Constant(0)
    assertions_obj_term = assertions_obj_term + concept_role_assertions_obj_term
    assertions_obj_term = assertions_obj_term + parameters['lambda_1'] * negated_concept_assertions_obj_term

    # Box-width regularization
    box_width_regularization_h_t, box_width_regularization_b = box_width_regularization_term(len(concepts), len(roles),
                                                                                             dimensionality, x_arr,
                                                                                             x_index_dict)
    assertions_obj_term = assertions_obj_term + parameters['lambda_2'] * box_width_regularization_h_t
    assertions_obj_term = assertions_obj_term + parameters['lambda_3'] * box_width_regularization_b

    objective_parts.append(assertions_obj_term)

    objective_scalar = cp.sum(cp.hstack(objective_parts))
    objective_function = cp.Minimize(objective_scalar)

    ### Constraints
    constraints.extend([x_arr[x_index_dict["constant_index"]] == 1])

    ###-###-### TBox Translation ###-###-###

    # Translating Concept Inclusions
    A_concept_inclusion = concept_inclusion_constraints(data["TBox"]['concepts'], dimensionality, x_arr, x_index_dict)
    if A_concept_inclusion is not None:
        constraints.extend([A_concept_inclusion @ x_arr <= 0])

    # # Translating Role Inclusions
    A_role_inclusion = role_inclusion_constraints(data["TBox"]['roles'], dimensionality, x_arr, x_index_dict)
    if A_role_inclusion is not None:
        constraints.extend([A_role_inclusion @ x_arr <= 0])

    ###### Box Consistency and Universe Constraints ######
    # Box Consistency
    A_box_consistency, b_box_consistency = box_consistency_constraints(concept_paths, dimensionality, x_arr,
                                                                       x_index_dict)
    if A_box_consistency is not None:
        constraints.append(A_box_consistency @ x_arr <= b_box_consistency)

    # Universe Constraints (Roles)
    A_role_width, b_role_width = role_box_width_constraints(roles, dimensionality, x_arr, x_index_dict)
    if A_role_width is not None:
        constraints.append(A_role_width @ x_arr <= b_role_width)

    # Universe Constraints (Concepts)
    A_concept_width, b_concept_width = concept_box_width_constraints(concepts, dimensionality, x_arr, x_index_dict)
    if A_concept_width is not None:
        constraints.append(A_concept_width @ x_arr <= b_concept_width)

    # Universe Constraints (Individuals)
    A_individual_bounds, b_individual_bounds = individual_bounds_constraints(individuals, dimensionality, x_arr,
                                                                             x_index_dict)
    if A_individual_bounds is not None:
        constraints.append(A_individual_bounds @ x_arr <= b_individual_bounds)

    problem = cp.Problem(objective_function, constraints)

    return problem

def signed_distance(x):
    for i in range(0, len(x)):
        if x[i] > 0:
            return np.linalg.norm(x[np.where(x > 0)])
    return np.max(x)


def box_distance(position, l, u):
    return signed_distance(np.hstack((l - position, position - u)))


def get_value_of_arr(var_arr):
    return var_arr.value


def score_concept_assertion(p_a, c_l, c_u):
    return -box_distance(p_a, c_l, c_u)


def score_role_assertion(hl, hu, tl, tu, bl, bu, pa, ba, pb, bb):
    return -1 * np.max([
        # hl: lower bound of the head box
        # hu: upper bound of the head box
        # tl: lower bound of the tail box
        # tu: upper bound of the tail box
        # bl: lower bound of bump box
        # bu: upper bound of bump box
        # pa: position of a
        # pb: position of b
        # ba: bump of a
        # bb: bump of b
        box_distance((pa + bb), hl, hu),
        box_distance((pb + ba), tl, tu),
        box_distance(ba, bl, bu),
        box_distance(bb, bl, bu),
    ]
    )


def getIndividual_position_eval(x_arr, x_index_dict, dimensionality, individual_name):
    start_index = x_index_dict["individuals"][individual_name]
    end_index = x_index_dict["individuals"][individual_name] + dimensionality
    return x_arr[start_index:end_index]


def getIndividual_bump_eval(x_arr, x_index_dict, dimensionality, individual_name):
    start_index = x_index_dict["individuals"][individual_name] + x_index_dict["individuals_bump_offset"]
    end_index = x_index_dict["individuals"][individual_name] + x_index_dict["individuals_bump_offset"] + dimensionality
    return x_arr[start_index:end_index]


def evaluate_f_vi_model(dataset, dimensionality, problem_parameter_dict, final_res_dir,
                        metrics, is_plot=False):
    evaluated_metrics = {}

    # x_arr are the variables in the model
    x_arr = x_variables_dict["x_arr"]
    # The mapping of the variable labels to their indices
    x_index_dict = x_variables_dict["x_index_dict"]

    for evaluation_mode in Utils.EVALUATION_MODES:
        data, concepts, roles, individuals, _, _, _ = loadDataset(dataset, evaluation_mode=evaluation_mode)

        # If there are no individuals then the concepts would start from index 0 in x_index_dict
        if x_index_dict["concepts_lb_offset"] == 0:
            return

        # individualVariables stores the different values that make up the different individuals to
        # decrease the number of calls and conversions needed for this process in later parts of the code
        # individualVariables: position -> Individual name -> position value; bump -> individual name -> bump value
        individualVariables = {
            "position": {},
            "bump": {}
        }

        for a in x_index_dict["individuals"].keys():
            # get_value_of_arr returns the value of the CVXPY variable
            # getIndividual_position_eval returns all the position CVXPY variables for an individual from x_arr according to its index in x_index_dict
            individualVariables["position"][a] = get_value_of_arr(
                getIndividual_position_eval(x_arr, x_index_dict, dimensionality, a))
            # getIndividual_bump_eval returns all the bump CVXPY variables for an individual from x_arr according to its index in x_index_dict
            individualVariables["bump"][a] = get_value_of_arr(
                getIndividual_bump_eval(x_arr, x_index_dict, dimensionality, a))

        parameter_dict = problem_parameter_dict.copy()
        parameter_dict['dimensionality'] = cp.Parameter(value=dimensionality, name='dimensionality', nonneg=True)

        concept_assertion_scores = {}

        for d in tqdm(x_index_dict["concepts"].keys()):
            c = Atomic_Concept(d)
            # getTransformedBox returns the lower and upper bounds of concept box as CVXPY variables
            l, u = getTransformedBox(c, dimensionality, x_arr, x_index_dict, True)

            # get_value_of_arr gets the values of the CVXPY variables
            c_l = get_value_of_arr(l)
            c_u = get_value_of_arr(u)
            # collects the scores of the different concepts
            concept_assertion_scores[c] = []
            for a in x_index_dict["individuals"].keys():
                p_a = individualVariables["position"][a]
                concept_assertion_scores[c] = concept_assertion_scores[c] + [
                    (a, score_concept_assertion(p_a, c_l, c_u))]

        # Sorting the scores of the concepts for ranking
        for d in tqdm(x_index_dict["concepts"].keys()):
            c = Atomic_Concept(d)
            concept_assertion_scores[c] = sorted(concept_assertion_scores[c], key=lambda d: -d[1].item())

        concept_ranks = {}
        # Computing the ranking
        for c in tqdm(concept_assertion_scores.keys()):
            i = 0
            concept_ranks[c] = []
            for a, s in concept_assertion_scores[c]:
                i = i + 1
                if (a, c) in data["Triples_To_Evaluate"]['concepts']:
                    concept_ranks[c] = concept_ranks[c] + [(a, i)]

        role_assertion_scores = {'head': {}, 'tail': {}}

        for ra in tqdm(data["Triples_To_Evaluate"]['roles']):
            x = ra[0]  # head
            r = ra[1].get_name()[0]  # role
            y = ra[2]  # tail

            # getTransformedBox returns the lower and upper bounds of each of he three boxes for the roles as CVXPY variables
            hl, hu, tl, tu, bl, bu = getTransformedBox(ra[1], dimensionality, x_arr, x_index_dict, True)

            # Get the position and bump variables for individual x (head)
            px = getIndividual_position_eval(x_arr, x_index_dict, dimensionality, x)
            bx = getIndividual_bump_eval(x_arr, x_index_dict, dimensionality, x)

            # Retrieve the values of x's pos and bump values (head)
            p_x = get_value_of_arr(px)
            b_x = get_value_of_arr(bx)

            # Get the position and bump variables for individual y (tail)
            py = getIndividual_position_eval(x_arr, x_index_dict, dimensionality, y)
            by = getIndividual_bump_eval(x_arr, x_index_dict, dimensionality, y)

            # Retrieve the values of y's pos and bump values (tail)
            p_y = get_value_of_arr(py)
            b_y = get_value_of_arr(by)

            # Retrieves the values of the different variables that make up the boundaries that form the boxes for the role r
            h_l = get_value_of_arr(hl)
            h_u = get_value_of_arr(hu)
            t_l = get_value_of_arr(tl)
            t_u = get_value_of_arr(tu)
            b_l = get_value_of_arr(bl)
            b_u = get_value_of_arr(bu)

            # role_assertion_scores shape:
            # 1. Head, Tail = size 2
            # 2. Number of queries (h, r, ?) = size is smaller than the number of the test triples because (h,r,?) pairs might occur multiple times e.g. (a, r, c) and (a, r, b)
            # 3. number of individuals = number of triples that occur in train and validation

            # Compute Head Scores
            role_assertion_scores['tail'][r, x] = []
            for b in x_index_dict["individuals"].keys():
                p_b = individualVariables["position"][b]
                b_b = individualVariables["bump"][b]
                role_assertion_scores['tail'][r, x] = role_assertion_scores['tail'][r, x] + [
                    (b, score_role_assertion(h_l, h_u, t_l, t_u, b_l, b_u, p_x, b_x, p_b, b_b))]

            # Compute Tail Scores
            role_assertion_scores['head'][r, y] = []
            for a in x_index_dict["individuals"].keys():
                p_a = individualVariables["position"][a]
                b_a = individualVariables["bump"][a]
                role_assertion_scores['head'][r, y] = role_assertion_scores['head'][r, y] + [
                    (a, score_role_assertion(h_l, h_u, t_l, t_u, b_l, b_u, p_a, b_a, p_y, b_y))]

        # Sort the scores of the roles to compute the ranking later on
        for ra in tqdm(data["Triples_To_Evaluate"]['roles']):
            x = ra[0]
            r = ra[1].get_name()[0]
            y = ra[2]

            role_assertion_scores['tail'][r, x] = sorted(role_assertion_scores['tail'][r, x],
                                                         key=lambda d: -d[1].item())
            role_assertion_scores['head'][r, y] = sorted(role_assertion_scores['head'][r, y],
                                                         key=lambda d: -d[1].item())

        # The filtering process of the ranking array to only include the evaluation triples
        # Filter_Assertions: The train and the validation sets
        # Triples_To_Evaluate: Test set

        role_ranks = {'head': {}, 'tail': {}}

        for (r, x) in tqdm(role_assertion_scores['tail'].keys()):
            i = 1
            role_ranks['tail'][(r, x)] = []
            for b, s in role_assertion_scores['tail'][(r, x)]:

                if (x, Atomic_Role(r), b) in data["Filter_Assertions"]['roles']:
                    # if x R b is in the train or validation set - then don't increase the rank
                    pass
                elif (x, Atomic_Role(r), b) in data["Triples_To_Evaluate"]['roles']:
                    # if x R b is in the test set - then return the rank
                    role_ranks['tail'][(r, x)] = role_ranks['tail'][(r, x)] + [(b, i)]
                else:
                    i = i + 1  # if xRb not in train, val, test, then increase the rank

        for (r, y) in tqdm(role_assertion_scores['head'].keys()):
            i = 1
            role_ranks['head'][(r, y)] = []
            for a, s in role_assertion_scores['head'][(r, y)]:
                if (a, Atomic_Role(r), y) in data["Filter_Assertions"]['roles']:
                    # if x R b is in the train or validation set - then don't increase the rank
                    pass
                elif (a, Atomic_Role(r), y) in data["Triples_To_Evaluate"]['roles']:
                    # if x R b is in the test set - then return the rank
                    role_ranks['head'][(r, y)] = role_ranks['head'][(r, y)] + [(a, i)]
                else:
                    i = i + 1  # if xRb not in train, val, test, then increase the rank

        filtered_role_ranks = []
        num_all_triples = 0
        for mode in ['head', 'tail']:
            for query in role_ranks[mode].keys():
                role_ranks_array = np.array([candidate_rank[1] for candidate_rank in role_ranks[mode][query]])
                num_all_triples = num_all_triples + role_ranks_array.size
                filtered_role_ranks.append(role_ranks_array)

        filtered_role_ranks = np.block(filtered_role_ranks)

        filtered_role_mrr = np.sum(1 / filtered_role_ranks) / num_all_triples
        filtered_hits_at_1 = np.sum(filtered_role_ranks <= 1) / num_all_triples
        filtered_hits_at_3 = np.sum(filtered_role_ranks <= 3) / num_all_triples
        filtered_hits_at_5 = np.sum(filtered_role_ranks <= 5) / num_all_triples
        filtered_hits_at_10 = np.sum(filtered_role_ranks <= 10) / num_all_triples

        evaluation_mode_text = ""

        if evaluation_mode == 'Test':
            print('\n\n ### Test Results ###')
            evaluation_mode_text = "test"
        elif evaluation_mode == 'Train':
            print('\n\n ### Train Results ###')
            evaluation_mode_text = "train"
        elif evaluation_mode == 'Validation':
            print('\n\n ### Validation Results ###')
            evaluation_mode_text = "validation"

        add_values_to_dict(evaluated_metrics, evaluation_mode_text,
                           {'MRR': filtered_role_mrr,
                            'Hits@1': filtered_hits_at_1,
                            'Hits@3': filtered_hits_at_3,
                            'Hits@5': filtered_hits_at_5,
                            'Hits@10': filtered_hits_at_10})

        print('filtered_mrr: ' + str(filtered_role_mrr))
        print('filtered_hits_at_1: ' + str(filtered_hits_at_1))
        print('filtered_hits_at_3: ' + str(filtered_hits_at_3))
        print('filtered_hits_at_5: ' + str(filtered_hits_at_5))
        print('filtered_hits_at_10: ' + str(filtered_hits_at_10))
        print('\n\n')

        final_res = {'concept_metrics': {},
                     'role_metrics': {'MRR': filtered_role_mrr,
                                      'Hits@1': filtered_hits_at_1,
                                      'Hits@3': filtered_hits_at_3,
                                      'Hits@5': filtered_hits_at_5,
                                      'Hits@10': filtered_hits_at_10}}

        reasoning_prefix = Utils.EVALUATION_PREFIXES[evaluation_mode]

        if not os.path.exists(final_res_dir):
            os.makedirs(final_res_dir)
        with open(final_res_dir + reasoning_prefix + 'result_table.tsv', 'a') as f:
            f.write('\n')
            for parameter_name in sorted_parameter_names:
                f.write(str(parameter_dict[parameter_name].value) + '\t')
            for metric in metrics:
                f.write(str(final_res['role_metrics'][metric]) + '\t')
            f.close()
    print(evaluated_metrics)

    if is_plot:
        if dataset == 'dummy':
            triples = data['ABox']['roles']
            import matplotlib.cm as cm

            complexConcepts = [
                Existential_Concept(Atomic_Role('r')),
                Neg_Concept(Existential_Concept(Atomic_Role('r'))),
                Existential_Concept(Atomic_Role('s')),
                Neg_Concept(Existential_Concept(Atomic_Role('s'))),
                Existential_Concept(Inverse_Role(Atomic_Role('s'))),
                Existential_Concept(Inverse_Role(Atomic_Role('r'))),
            ]

            # Auto-generate colors and labels
            colors = cm.tab20(np.linspace(0, 1, len(complexConcepts)))
            auto_colorMap = {}
            auto_labelMap = {}

            for i, c in enumerate(complexConcepts):
                c_name = c.get_op_name()
                auto_colorMap[c_name] = colors[i]
                auto_labelMap[c_name] = c_name

            # Merge with existing maps
            colorMap = {**auto_colorMap, **Utils.CONCEPT_COLOR_MAP}
            labelMap = {**auto_labelMap, **Utils.LABEL_MAP}

            plot_and_save_solution_weak(
                complexConcepts,
                [],
                label_positions=None,
                axis_limits=Utils.AXIS_LIMITS,
                label_map=labelMap,
                colorMap=colorMap,
                dimensionality=dimensionality,
                x_arr=x_arr,
                x_index_dict=x_index_dict,
                plot_file_name="dummy",
                individuals_positions=individualVariables["position"],
                individuals_bumps=individualVariables["bump"],
                triples=triples,
                interactive=True
            )
        else:
            triples = data['ABox']['roles']
            plot_and_save_solution_weak(
                [
                    Existential_Concept(Atomic_Role('hasSibling')),
                    Existential_Concept(Inverse_Role(Atomic_Role('hasSibling'))),
                    Existential_Concept(Atomic_Role('hasChild')),
                    Existential_Concept(Inverse_Role(Atomic_Role('hasChild'))),

                    Existential_Concept(Atomic_Role('hasMother')),
                    Existential_Concept(Inverse_Role(Atomic_Role('hasMother'))),
                    Existential_Concept(Atomic_Role('hasFather')),
                    Existential_Concept(Inverse_Role(Atomic_Role('hasFather'))),
                    Existential_Concept(Atomic_Role('hasParent')),
                    Existential_Concept(Inverse_Role(Atomic_Role('hasParent'))),

                    Existential_Concept(Atomic_Role('spouse')),
                    Existential_Concept(Inverse_Role(Atomic_Role('spouse'))),
                    Existential_Concept(Atomic_Role('relative')),
                    Existential_Concept(Inverse_Role(Atomic_Role('relative'))),
                ], [],
                label_positions=None,
                axis_limits=Utils.AXIS_LIMITS,
                label_map=Utils.LABEL_MAP,
                colorMap=Utils.CONCEPT_COLOR_MAP, dimensionality=dimensionality,
                x_arr=x_arr, x_index_dict=x_index_dict, plot_file_name="test",
                individuals_positions=individualVariables["position"],
                individuals_bumps=individualVariables["bump"],
                triples=triples,
                plot_axis=False
            )
    return evaluated_metrics


def add_values_to_dict(dict_arr, term, value):
    if term not in dict_arr:
        dict_arr[term] = {}
    dict_arr[term] = value


def parse_kwargs(**kwargs):
    if 'config' in kwargs.keys():
        config_path = kwargs['config']
        # print(config_path)
        with open(config_path, "r") as f:
            config = json.loads(f.read())
    else:
        raise Exception("No config input file!")

    if 'plot' in kwargs.keys():
        val = kwargs['plot']
        if val == 'true':
            plot_flag = True
        elif val == 'false':
            plot_flag = False
        else:
            raise Exception("No value for the <plot> parameter provided!")
    else:
        plot_flag = False
    return config, plot_flag


def convex_BoxLitE_solver(dataset, exp_name, dimensionality, parameters, coefficients, solver_name_var, plot_flag):
    final_res_dir = 'Benchmarking/BoxLitE/' + exp_name
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']

    if not os.path.exists(final_res_dir):
        os.makedirs(final_res_dir)

    for reasoning_prefix in Utils.EVALUATION_PREFIXES.values():
        with open(final_res_dir + reasoning_prefix + 'result_table.tsv', 'a') as f:
            for parameter in sorted_parameter_names:
                f.write(parameter)
                f.write('\t')
            for metric in metrics:
                f.write(metric)
                f.write('\t')
    problem = (create_Problem(dataset, dimensionality, parameters))
    problem.param_dict['lambda_1'].value = coefficients['lambda_1']
    problem.param_dict['lambda_2'].value = coefficients['lambda_2']
    problem.param_dict['lambda_3'].value = coefficients['lambda_3']

    problem.solve(solver=solver_name_var, verbose=True)
    print("prob.status:", problem.status)
    print("prob.value:", problem.value)
    print("objective value via expr:", problem.objective.value)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        evaluate_f_vi_model(dataset, dimensionality,
                            problem.param_dict,
                            final_res_dir, metrics, plot_flag)


def parameter_construction():
    lambda_1 = cp.Parameter(value=0,
                            name='lambda_1',
                            nonneg=True)
    lambda_2 = cp.Parameter(value=0, name='lambda_2', nonneg=True)
    lambda_3 = cp.Parameter(value=0, name='lambda_3', nonneg=True)

    parameters = {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_3": lambda_3,
    }

    return parameters


def convex_BoxLitE_exec_parameterized(experiment_name, dataset, S_OMEGA, dimensionality,
                                      solver, lambda_1, lambda_2, lambda_3, plot_flag=False):
    Utils.S_OMEGA = S_OMEGA
    parameters = parameter_construction()

    coefficients = {
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_3': lambda_3,
    }

    convex_BoxLitE_solver(dataset, experiment_name, dimensionality,
                          parameters, coefficients, solver, plot_flag)


def convex_BoxLitE_exec(**kwargs):
    config, plot_flag = parse_kwargs(**kwargs)

    experiment_name = config['exp_name']
    dataset = config['dataset']
    S_OMEGA = config['s_omega']
    dimensionality = config['dimensionality']
    lambda_1 = config['lambda_1']
    lambda_2 = config['lambda_2']
    lambda_3 = config['lambda_3']
    convex_BoxLitE_exec_parameterized(experiment_name=experiment_name, dataset=dataset,
                                      S_OMEGA=S_OMEGA, dimensionality=dimensionality, solver=cp.MOSEK,
                                      lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,
                                      plot_flag=plot_flag)


if __name__ == '__main__':
    convex_BoxLitE_exec(**dict(arg.split('=') for arg in sys.argv[1:]))
