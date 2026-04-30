import numpy as np
import cvxpy as cp
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from DL_Lite_Model.Neg_Concept import Neg_Concept
from DL_Lite_Model.Existential_Concept import Existential_Concept

def getTransformedBox(complex_concept_or_role, dimensionality, x_arr, x_index_dict, values=False,
                      existential_flag=False):
    name, type = complex_concept_or_role.get_name()

    if type == "concept":
        col_l = x_index_dict["concepts"][name] + x_index_dict["concepts_lb_relative_offset"]
        col_u = x_index_dict["concepts"][name] + x_index_dict["concepts_ub_relative_offset"]
        params = complex_concept_or_role.apply_ops({'col_l': np.concatenate(
            [np.array(np.array(range(dimensionality)) + col_l), np.array([x_index_dict['constant_index']])]),
            'data_l': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_u': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_u),
                 np.array([x_index_dict['constant_index']])]),
            'data_u': np.concatenate([np.ones(dimensionality), np.array([0])])})
    elif type == 'role':
        col_hl = x_index_dict["roles"][name] + x_index_dict["roles_head_lb_relative_offset"]
        col_hu = x_index_dict["roles"][name] + x_index_dict["roles_head_ub_relative_offset"]
        col_tl = x_index_dict["roles"][name] + x_index_dict["roles_tail_lb_relative_offset"]
        col_tu = x_index_dict["roles"][name] + x_index_dict["roles_tail_ub_relative_offset"]
        col_bl = x_index_dict["roles"][name] + x_index_dict["roles_bump_lb_relative_offset"]
        col_bu = x_index_dict["roles"][name] + x_index_dict["roles_bump_ub_relative_offset"]
        params = complex_concept_or_role.apply_ops({
            'col_hl': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_hl), np.array([x_index_dict['constant_index']])]),
            'data_hl': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_hu': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_hu), np.array([x_index_dict['constant_index']])]),
            'data_hu': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_tl': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_tl), np.array([x_index_dict['constant_index']])]),
            'data_tl': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_tu': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_tu), np.array([x_index_dict['constant_index']])]),
            'data_tu': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_bl': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_bl), np.array([x_index_dict['constant_index']])]),
            'data_bl': np.concatenate([np.ones(dimensionality), np.array([0])]),
            'col_bu': np.concatenate(
                [np.array(np.array(range(dimensionality)) + col_bu), np.array([x_index_dict['constant_index']])]),
            'data_bu': np.concatenate([np.ones(dimensionality), np.array([0])])})
    else:
        raise Exception("Axioms can only be based on roles or concepts!")

    if not values:
        return params
    else:
        # Return the values of the parameters
        if type == "concept" or isinstance(complex_concept_or_role, Existential_Concept) \
                or isinstance(complex_concept_or_role, Neg_Concept):

            l = cp.multiply(x_arr[params['col_l'][:-1]],
                            params['data_l'][:-1])  # get the variable entries in x, multiplied by the data
            c = cp.multiply(x_arr[params['col_l'][-1]],
                            params['data_l'][-1])  # get the constant entry in x, multiplied by the data
            l = l + c

            u = cp.multiply(x_arr[params['col_u'][:-1]], params['data_u'][
                :-1])  # get the variable entries in x, multiplied by the data
            c = cp.multiply(x_arr[params['col_u'][-1]], params['data_u'][
                -1])  # get the constant entry in x, multiplied by the data
            u = u + c
            if existential_flag:
                existential_nested = False
                concept_copy = copy.deepcopy(complex_concept_or_role)
                while True:
                    if isinstance(concept_copy, Existential_Concept):
                        existential_nested = True
                        break

                    if hasattr(concept_copy, 'concept'):
                        concept_copy = concept_copy.concept
                    else:
                        break
                if existential_nested:
                    mid_index = dimensionality
                    l = l[0:mid_index] - l[mid_index:]
                    u = u[0:mid_index] - u[mid_index:]

            return l, u
        elif type == 'role':
            hl = cp.multiply(x_arr[params['col_hl'][:-1]], params['data_hl'][:-1])
            c = cp.multiply(x_arr[params['col_hl'][-1]], params['data_hl'][-1])
            hl = hl + c

            hu = cp.multiply(x_arr[params['col_hu'][:-1]], params['data_hu'][:-1])
            c = cp.multiply(x_arr[params['col_hu'][-1]], params['data_hu'][-1])
            hu = hu + c

            tl = cp.multiply(x_arr[params['col_tl'][:-1]], params['data_tl'][:-1])
            c = cp.multiply(x_arr[params['col_tl'][-1]], params['data_tl'][-1])
            tl = tl + c

            tu = cp.multiply(x_arr[params['col_tu'][:-1]], params['data_tu'][:-1])
            c = cp.multiply(x_arr[params['col_tu'][-1]], params['data_tu'][-1])
            tu = tu + c

            bl = cp.multiply(x_arr[params['col_bl'][:-1]], params['data_bl'][:-1])
            c = cp.multiply(x_arr[params['col_bl'][-1]], params['data_bl'][-1])
            bl = bl + c

            bu = cp.multiply(x_arr[params['col_bu'][:-1]], params['data_bu'][:-1])
            c = cp.multiply(x_arr[params['col_bu'][-1]], params['data_bu'][-1])
            bu = bu + c

            return hl, hu, tl, tu, bl, bu


def getCenter(l, u):
    return (u + l) / 2

def create_3d_box(l, u, color, alpha=0.3):
    """Create a 3D box from lower and upper bounds."""
    vertices = [
        [l[0], l[1], l[2]],
        [u[0], l[1], l[2]],
        [u[0], u[1], l[2]],
        [l[0], u[1], l[2]],
        [l[0], l[1], u[2]],
        [u[0], l[1], u[2]],
        [u[0], u[1], u[2]],
        [l[0], u[1], u[2]]
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    return Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=1)


def plot_and_save_solution_interactive(complexConcepts, complexRoles,
                                       label_map, colorMap, dimensionality, x_arr, x_index_dict,
                                       plot_file_name, individuals_positions=None, individuals_bumps=None,
                                       triples=None):
    if individuals_positions is None:
        individuals_positions = {}
    if triples is None:
        triples = []

    offset = 0.02

    # Track current mode and dimensions
    state = {
        'mode': '2D',
        'dim_x': 0,
        'dim_y': 1,
        'dim_z': 2,
        'zoom_level': 1.0
    }
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.3, left=0.1)

    # Initial 2D axis
    ax = fig.add_subplot(111)

    def get_plot_data_2d(dim_x, dim_y):
        """Extract data for the selected 2D dimensions."""
        plot_data = {
            'individuals': [],
            'head_tail_points': [],
            'concepts': [],
            'roles': [],
            'all_x': [],
            'all_y': []
        }

        if individuals_positions:
            for point in individuals_positions:
                x = individuals_positions[point][dim_x]
                y = individuals_positions[point][dim_y]
                plot_data['individuals'].append((point, x, y))
                plot_data['all_x'].append(x)
                plot_data['all_y'].append(y)

            for triple in triples:
                i1 = triple[0]
                i2 = triple[2]
                xh = individuals_positions[i1][dim_x] + individuals_bumps[i2][dim_x]
                yh = individuals_positions[i1][dim_y] + individuals_bumps[i2][dim_y]
                xt = individuals_positions[i2][dim_x] + individuals_bumps[i1][dim_x]
                yt = individuals_positions[i2][dim_y] + individuals_bumps[i1][dim_y]
                plot_data['head_tail_points'].append((i1, i2, xh, yh, xt, yt))
                plot_data['all_x'].extend([xh, xt])
                plot_data['all_y'].extend([yh, yt])

        for c in complexConcepts:
            c_name = c.get_op_name()
            l, u = getTransformedBox(c, dimensionality, x_arr, x_index_dict, True)
            l = l.value
            u = u.value

            lx, ly = l[dim_x], l[dim_y]
            ux, uy = u[dim_x], u[dim_y]

            plot_data['concepts'].append({
                'name': c_name,
                'l': (lx, ly),
                'u': (ux, uy),
                'color': colorMap.get(c_name, 'blue')
            })
            plot_data['all_x'].extend([lx, ux])
            plot_data['all_y'].extend([ly, uy])

        for r in complexRoles:
            r_name = r.get_op_name()
            hl, hu, tl, tu, bl, bu = getTransformedBox(r, dimensionality, x_arr, x_index_dict, True)

            for pos, (l_vec, u_vec) in {'h': (hl, hu), 't': (tl, tu)}.items():
                r_name_pos = r_name + "_" + pos
                l = l_vec.value
                u = u_vec.value

                lx, ly = l[dim_x], l[dim_y]
                ux, uy = u[dim_x], u[dim_y]

                plot_data['roles'].append({
                    'name': r_name_pos,
                    'l': (lx, ly),
                    'u': (ux, uy),
                    'color': colorMap.get(r_name_pos, 'green')
                })
                plot_data['all_x'].extend([lx, ux])
                plot_data['all_y'].extend([ly, uy])

        return plot_data

    def get_plot_data_3d(dim_x, dim_y, dim_z):
        """Extract data for the selected 3D dimensions."""
        plot_data = {
            'individuals': [],
            'head_tail_points': [],
            'concepts': [],
            'roles': [],
            'all_x': [],
            'all_y': [],
            'all_z': []
        }

        if individuals_positions:
            for point in individuals_positions:
                x = individuals_positions[point][dim_x]
                y = individuals_positions[point][dim_y]
                z = individuals_positions[point][dim_z]
                plot_data['individuals'].append((point, x, y, z))
                plot_data['all_x'].append(x)
                plot_data['all_y'].append(y)
                plot_data['all_z'].append(z)

            for triple in triples:
                i1 = triple[0]
                i2 = triple[2]
                xh = individuals_positions[i1][dim_x] + individuals_bumps[i2][dim_x]
                yh = individuals_positions[i1][dim_y] + individuals_bumps[i2][dim_y]
                zh = individuals_positions[i1][dim_z] + individuals_bumps[i2][dim_z]
                xt = individuals_positions[i2][dim_x] + individuals_bumps[i1][dim_x]
                yt = individuals_positions[i2][dim_y] + individuals_bumps[i1][dim_y]
                zt = individuals_positions[i2][dim_z] + individuals_bumps[i1][dim_z]
                plot_data['head_tail_points'].append((i1, i2, xh, yh, zh, xt, yt, zt))
                plot_data['all_x'].extend([xh, xt])
                plot_data['all_y'].extend([yh, yt])
                plot_data['all_z'].extend([zh, zt])

        for c in complexConcepts:
            c_name = c.get_op_name()
            l, u = getTransformedBox(c, dimensionality, x_arr, x_index_dict, True)
            l = l.value
            u = u.value

            plot_data['concepts'].append({
                'name': c_name,
                'l': (l[dim_x], l[dim_y], l[dim_z]),
                'u': (u[dim_x], u[dim_y], u[dim_z]),
                'color': colorMap.get(c_name, 'blue')
            })
            plot_data['all_x'].extend([l[dim_x], u[dim_x]])
            plot_data['all_y'].extend([l[dim_y], u[dim_y]])
            plot_data['all_z'].extend([l[dim_z], u[dim_z]])

        for r in complexRoles:
            r_name = r.get_op_name()
            hl, hu, tl, tu, bl, bu = getTransformedBox(r, dimensionality, x_arr, x_index_dict, True)

            for pos, (l_vec, u_vec) in {'h': (hl, hu), 't': (tl, tu)}.items():
                r_name_pos = r_name + "_" + pos
                l = l_vec.value
                u = u_vec.value

                plot_data['roles'].append({
                    'name': r_name_pos,
                    'l': (l[dim_x], l[dim_y], l[dim_z]),
                    'u': (u[dim_x], u[dim_y], u[dim_z]),
                    'color': colorMap.get(r_name_pos, 'green')
                })
                plot_data['all_x'].extend([l[dim_x], u[dim_x]])
                plot_data['all_y'].extend([l[dim_y], u[dim_y]])
                plot_data['all_z'].extend([l[dim_z], u[dim_z]])

        return plot_data

    def focus_view_2d(data):
        """Calculate tight bounds around the boxes (only for 2D)."""
        box_x = []
        box_y = []

        # Only include concept boxes
        for concept in data['concepts']:
            box_x.extend([concept['l'][0], concept['u'][0]])
            box_y.extend([concept['l'][1], concept['u'][1]])

        # Only include role boxes
        for role in data['roles']:
            box_x.extend([role['l'][0], role['u'][0]])
            box_y.extend([role['l'][1], role['u'][1]])

        if not box_x or not box_y:
            return None, None, None, None

        x_min, x_max = min(box_x), max(box_x)
        y_min, y_max = min(box_y), max(box_y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        # If the range is very small, expand it to make boxes visible
        # Use the center and create a reasonable viewing window
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Minimum visible range - at least 10x the data range, or a small absolute value
        min_visible_range_x = max(x_range * 10, abs(x_center) * 0.001, 1e-10)
        min_visible_range_y = max(y_range * 10, abs(y_center) * 0.001, 1e-10)

        # Use the larger of actual range or minimum visible range
        half_range_x = max(x_range / 2, min_visible_range_x / 2)
        half_range_y = max(y_range / 2, min_visible_range_y / 2)

        # Small margin (5%)
        margin_x = half_range_x * 0.05
        margin_y = half_range_y * 0.05

        return (x_center - half_range_x - margin_x,
                x_center + half_range_x + margin_x,
                y_center - half_range_y - margin_y,
                y_center + half_range_y + margin_y)

    def focus_view_3d(data):
        """Calculate tight bounds around the boxes (only for 3D)."""
        box_x = []
        box_y = []
        box_z = []

        # Only include concept boxes
        for concept in data['concepts']:
            box_x.extend([concept['l'][0], concept['u'][0]])
            box_y.extend([concept['l'][1], concept['u'][1]])
            box_z.extend([concept['l'][2], concept['u'][2]])

        # Only include role boxes
        for role in data['roles']:
            box_x.extend([role['l'][0], role['u'][0]])
            box_y.extend([role['l'][1], role['u'][1]])
            box_z.extend([role['l'][2], role['u'][2]])

        if not box_x or not box_y or not box_z:
            return None

        x_min, x_max = min(box_x), max(box_x)
        y_min, y_max = min(box_y), max(box_y)
        z_min, z_max = min(box_z), max(box_z)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Minimum visible range
        min_visible_range_x = max(x_range * 10, abs(x_center) * 0.001, 1e-10)
        min_visible_range_y = max(y_range * 10, abs(y_center) * 0.001, 1e-10)
        min_visible_range_z = max(z_range * 10, abs(z_center) * 0.001, 1e-10)

        half_range_x = max(x_range / 2, min_visible_range_x / 2)
        half_range_y = max(y_range / 2, min_visible_range_y / 2)
        half_range_z = max(z_range / 2, min_visible_range_z / 2)

        margin_x = half_range_x * 0.05
        margin_y = half_range_y * 0.05
        margin_z = half_range_z * 0.05

        return {
            'x': (x_center - half_range_x - margin_x, x_center + half_range_x + margin_x),
            'y': (y_center - half_range_y - margin_y, y_center + half_range_y + margin_y),
            'z': (z_center - half_range_z - margin_z, z_center + half_range_z + margin_z)
        }

    def draw_2d(dim_x, dim_y, auto_focus=False):
        """Draw 2D plot."""
        nonlocal ax
        fig.clear()
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.3, left=0.1)

        data = get_plot_data_2d(dim_x, dim_y)

        for point, x, y in data['individuals']:
            ax.scatter(x, y, c='black', s=50, zorder=5)
            ax.text(x + offset, y + offset, point, color='black', fontsize=8)

        for i1, i2, xh, yh, xt, yt in data['head_tail_points']:
            ax.scatter(xh, yh, c='orange', s=50, zorder=5)
            ax.scatter(xt, yt, c='purple', s=50, zorder=5)
            ax.text(xh + offset, yh + offset, i1, color='orange', fontsize=8)
            ax.text(xt + offset, yt + offset, i2, color='purple', fontsize=8)

        for concept in data['concepts']:
            lx, ly = concept['l']
            ux, uy = concept['u']

            rect = patches.Rectangle(
                (lx, ly), ux - lx, uy - ly,
                linewidth=1, edgecolor=concept['color'], facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)

            label_text = label_map.get(concept['name'], concept['name'])
            ax.text((lx + ux) / 2, uy + offset * 2, label_text, color=concept['color'], fontsize=9)

        for role in data['roles']:
            lx, ly = role['l']
            ux, uy = role['u']

            rect = patches.Rectangle(
                (lx, ly), ux - lx, uy - ly,
                linewidth=1, edgecolor=role['color'], facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)

            label_text = label_map.get(role['name'], role['name'])
            ax.text((lx + ux) / 2, uy + offset * 2, label_text, color=role['color'], fontsize=9)

        # Apply focus if requested
        if auto_focus:
            x_min, x_max, y_min, y_max = focus_view_2d(data)
            if x_min is not None:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        else:
            # Default: show all data with margin
            if data['all_x'] and data['all_y']:
                x_margin = (max(data['all_x']) - min(data['all_x'])) * 0.1 + 0.1
                y_margin = (max(data['all_y']) - min(data['all_y'])) * 0.1 + 0.1
                ax.set_xlim(min(data['all_x']) - x_margin, max(data['all_x']) + x_margin)
                ax.set_ylim(min(data['all_y']) - y_margin, max(data['all_y']) + y_margin)

        ax.set_xlabel(f'Dimension {dim_x}')
        ax.set_ylabel(f'Dimension {dim_y}')
        ax.set_title(f'2D View: Dim {dim_x} vs Dim {dim_y}')
        ax.set_aspect('equal', adjustable='box')

        recreate_widgets()
        fig.canvas.draw_idle()

    def draw_3d(dim_x, dim_y, dim_z, auto_focus=False):
        """Draw 3D plot."""
        nonlocal ax
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.3, left=0.1)

        data = get_plot_data_3d(dim_x, dim_y, dim_z)

        for point, x, y, z in data['individuals']:
            ax.scatter(x, y, z, c='black', s=50, zorder=5)
            ax.text(x + offset, y + offset, z + offset, point, color='black', fontsize=8)

        for i1, i2, xh, yh, zh, xt, yt, zt in data['head_tail_points']:
            ax.scatter(xh, yh, zh, c='orange', s=50, zorder=5)
            ax.scatter(xt, yt, zt, c='purple', s=50, zorder=5)
            ax.text(xh + offset, yh + offset, zh + offset, i1, color='orange', fontsize=8)
            ax.text(xt + offset, yt + offset, zt + offset, i2, color='purple', fontsize=8)

        for concept in data['concepts']:
            box = create_3d_box(concept['l'], concept['u'], concept['color'], alpha=0.2)
            ax.add_collection3d(box)

            label_text = label_map.get(concept['name'], concept['name'])
            cx = (concept['l'][0] + concept['u'][0]) / 2
            cy = (concept['l'][1] + concept['u'][1]) / 2
            cz = concept['u'][2] + offset
            ax.text(cx, cy, cz, label_text, color=concept['color'], fontsize=9)

        for role in data['roles']:
            box = create_3d_box(role['l'], role['u'], role['color'], alpha=0.2)
            ax.add_collection3d(box)

            label_text = label_map.get(role['name'], role['name'])
            cx = (role['l'][0] + role['u'][0]) / 2
            cy = (role['l'][1] + role['u'][1]) / 2
            cz = role['u'][2] + offset
            ax.text(cx, cy, cz, label_text, color=role['color'], fontsize=9)

        # Apply focus if requested
        if auto_focus:
            bounds = focus_view_3d(data)
            if bounds:
                ax.set_xlim(bounds['x'])
                ax.set_ylim(bounds['y'])
                ax.set_zlim(bounds['z'])
        else:
            # Default: show all data with margin
            if data['all_x'] and data['all_y'] and data['all_z']:
                margin = 0.1
                x_range = max(data['all_x']) - min(data['all_x'])
                y_range = max(data['all_y']) - min(data['all_y'])
                z_range = max(data['all_z']) - min(data['all_z'])
                ax.set_xlim(min(data['all_x']) - x_range * margin, max(data['all_x']) + x_range * margin)
                ax.set_ylim(min(data['all_y']) - y_range * margin, max(data['all_y']) + y_range * margin)
                ax.set_zlim(min(data['all_z']) - z_range * margin, max(data['all_z']) + z_range * margin)

        ax.set_xlabel(f'Dimension {dim_x}')
        ax.set_ylabel(f'Dimension {dim_y}')
        ax.set_zlabel(f'Dimension {dim_z}')
        ax.set_title(f'3D View: Dim {dim_x} vs Dim {dim_y} vs Dim {dim_z}')

        recreate_widgets()
        fig.canvas.draw_idle()

    # Widget references
    widgets = {}

    def recreate_widgets():
        """Recreate all widgets after figure clear."""
        # Mode selector
        ax_mode = plt.axes([0.02, 0.15, 0.1, 0.1])
        widgets['radio'] = RadioButtons(ax_mode, ('2D', '3D'), active=0 if state['mode'] == '2D' else 1)
        widgets['radio'].on_clicked(on_mode_change)

        # Dimension sliders
        ax_dim_x = plt.axes([0.2, 0.15, 0.35, 0.03])
        ax_dim_y = plt.axes([0.2, 0.10, 0.35, 0.03])
        ax_dim_z = plt.axes([0.2, 0.05, 0.35, 0.03])

        widgets['slider_x'] = Slider(ax_dim_x, 'X Dim', 0, dimensionality - 1, valinit=state['dim_x'], valstep=1)
        widgets['slider_y'] = Slider(ax_dim_y, 'Y Dim', 0, dimensionality - 1, valinit=state['dim_y'], valstep=1)
        widgets['slider_z'] = Slider(ax_dim_z, 'Z Dim', 0, dimensionality - 1, valinit=state['dim_z'], valstep=1)

        widgets['slider_x'].on_changed(on_slider_change)
        widgets['slider_y'].on_changed(on_slider_change)
        widgets['slider_z'].on_changed(on_slider_change)

        if state['mode'] == '2D':
            widgets['slider_z'].ax.set_visible(False)

        # === Buttons Section ===
        # Row 1: Focus and Save
        ax_focus = plt.axes([0.6, 0.15, 0.08, 0.04])
        widgets['btn_focus'] = Button(ax_focus, 'Focus')
        widgets['btn_focus'].on_clicked(on_focus)

        ax_save = plt.axes([0.69, 0.15, 0.08, 0.04])
        widgets['btn_save'] = Button(ax_save, 'Save')
        widgets['btn_save'].on_clicked(save_plot)

        # Row 2: Zoom controls
        ax_zoom_in = plt.axes([0.6, 0.10, 0.08, 0.04])
        widgets['btn_zoom_in'] = Button(ax_zoom_in, 'Zoom +')
        widgets['btn_zoom_in'].on_clicked(on_zoom_in)

        ax_zoom_out = plt.axes([0.69, 0.10, 0.08, 0.04])
        widgets['btn_zoom_out'] = Button(ax_zoom_out, 'Zoom -')
        widgets['btn_zoom_out'].on_clicked(on_zoom_out)

        # Reset button
        ax_reset = plt.axes([0.78, 0.12, 0.08, 0.04])
        widgets['btn_reset'] = Button(ax_reset, 'Reset')
        widgets['btn_reset'].on_clicked(on_reset)

    def on_reset(event):
        state['dim_x'] = 0
        state['dim_y'] = 1
        state['dim_z'] = 2
        if state['mode'] == '2D':
            draw_2d(state['dim_x'], state['dim_y'], auto_focus=False)
        else:
            draw_3d(state['dim_x'], state['dim_y'], state['dim_z'], auto_focus=False)

    def on_zoom_in(event):
        """Zoom in by 2x centered on current view."""
        if state['mode'] == '2D':
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_half = (x_max - x_min) / 4  # Divide by 4 = zoom 2x
            y_half = (y_max - y_min) / 4

            ax.set_xlim(x_center - x_half, x_center + x_half)
            ax.set_ylim(y_center - y_half, y_center + y_half)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            x_half = (x_max - x_min) / 4
            y_half = (y_max - y_min) / 4
            z_half = (z_max - z_min) / 4

            ax.set_xlim(x_center - x_half, x_center + x_half)
            ax.set_ylim(y_center - y_half, y_center + y_half)
            ax.set_zlim(z_center - z_half, z_center + z_half)

        fig.canvas.draw_idle()

    def on_zoom_out(event):
        """Zoom out by 2x centered on current view."""
        if state['mode'] == '2D':
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_half = (x_max - x_min)  # Multiply by 2 = zoom out 2x
            y_half = (y_max - y_min)

            ax.set_xlim(x_center - x_half, x_center + x_half)
            ax.set_ylim(y_center - y_half, y_center + y_half)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            x_half = (x_max - x_min)
            y_half = (y_max - y_min)
            z_half = (z_max - z_min)

            ax.set_xlim(x_center - x_half, x_center + x_half)
            ax.set_ylim(y_center - y_half, y_center + y_half)
            ax.set_zlim(z_center - z_half, z_center + z_half)

        fig.canvas.draw_idle()

    def on_mode_change(label):
        state['mode'] = label
        if label == '2D':
            draw_2d(state['dim_x'], state['dim_y'], auto_focus=False)
        else:
            draw_3d(state['dim_x'], state['dim_y'], state['dim_z'], auto_focus=False)

    def on_slider_change(val):
        state['dim_x'] = int(widgets['slider_x'].val)
        state['dim_y'] = int(widgets['slider_y'].val)
        state['dim_z'] = int(widgets['slider_z'].val)

        if state['mode'] == '2D':
            draw_2d(state['dim_x'], state['dim_y'], auto_focus=False)
        else:
            draw_3d(state['dim_x'], state['dim_y'], state['dim_z'], auto_focus=False)

    def on_focus(event):
        """Focus button handler - zoom to fit all data."""
        if state['mode'] == '2D':
            data = get_plot_data_2d(state['dim_x'], state['dim_y'])
            x_min, x_max, y_min, y_max = focus_view_2d(data)
            if x_min is not None:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                fig.canvas.draw_idle()
        else:
            data = get_plot_data_3d(state['dim_x'], state['dim_y'], state['dim_z'])
            bounds = focus_view_3d(data)
            if bounds:
                ax.set_xlim(bounds['x'])
                ax.set_ylim(bounds['y'])
                ax.set_zlim(bounds['z'])
                fig.canvas.draw_idle()

    def save_plot(event):
        if state['mode'] == '2D':
            save_fig, save_ax = plt.subplots(figsize=(10, 8))

            data = get_plot_data_2d(state['dim_x'], state['dim_y'])

            for point, x, y in data['individuals']:
                save_ax.scatter(x, y, c='black', s=50, zorder=5)
                save_ax.text(x + offset, y + offset, point, color='black', fontsize=8)

            for i1, i2, xh, yh, xt, yt in data['head_tail_points']:
                save_ax.scatter(xh, yh, c='orange', s=50, zorder=5)
                save_ax.scatter(xt, yt, c='purple', s=50, zorder=5)
                save_ax.text(xh + offset, yh + offset, i1, color='orange', fontsize=8)
                save_ax.text(xt + offset, yt + offset, i2, color='purple', fontsize=8)

            for concept in data['concepts']:
                lx, ly = concept['l']
                ux, uy = concept['u']

                rect = patches.Rectangle(
                    (lx, ly), ux - lx, uy - ly,
                    linewidth=1, edgecolor=concept['color'], facecolor='none', alpha=0.8
                )
                save_ax.add_patch(rect)

                label_text = label_map.get(concept['name'], concept['name'])
                save_ax.text((lx + ux) / 2, uy + offset * 2, label_text, color=concept['color'], fontsize=9)

            for role in data['roles']:
                lx, ly = role['l']
                ux, uy = role['u']

                rect = patches.Rectangle(
                    (lx, ly), ux - lx, uy - ly,
                    linewidth=1, edgecolor=role['color'], facecolor='none', alpha=0.8
                )
                save_ax.add_patch(rect)

                label_text = label_map.get(role['name'], role['name'])
                save_ax.text((lx + ux) / 2, uy + offset * 2, label_text, color=role['color'], fontsize=9)

            # Copy current axis limits from interactive view
            save_ax.set_xlim(ax.get_xlim())
            save_ax.set_ylim(ax.get_ylim())

            save_ax.set_xlabel(f'Dimension {state["dim_x"]}')
            save_ax.set_ylabel(f'Dimension {state["dim_y"]}')
            save_ax.set_title(f'2D View: Dim {state["dim_x"]} vs Dim {state["dim_y"]}')
            save_ax.set_aspect('equal', adjustable='box')

            filename = f"Plots/{plot_file_name}_2D_dim{state['dim_x']}_dim{state['dim_y']}.svg"

        else:  # 3D
            save_fig = plt.figure(figsize=(10, 8))
            save_ax = save_fig.add_subplot(111, projection='3d')

            data = get_plot_data_3d(state['dim_x'], state['dim_y'], state['dim_z'])

            for point, x, y, z in data['individuals']:
                save_ax.scatter(x, y, z, c='black', s=50, zorder=5)
                save_ax.text(x + offset, y + offset, z + offset, point, color='black', fontsize=8)

            for i1, i2, xh, yh, zh, xt, yt, zt in data['head_tail_points']:
                save_ax.scatter(xh, yh, zh, c='orange', s=50, zorder=5)
                save_ax.scatter(xt, yt, zt, c='purple', s=50, zorder=5)
                save_ax.text(xh + offset, yh + offset, zh + offset, i1, color='orange', fontsize=8)
                save_ax.text(xt + offset, yt + offset, zt + offset, i2, color='purple', fontsize=8)

            for concept in data['concepts']:
                box = create_3d_box(concept['l'], concept['u'], concept['color'], alpha=0.2)
                save_ax.add_collection3d(box)

                label_text = label_map.get(concept['name'], concept['name'])
                cx = (concept['l'][0] + concept['u'][0]) / 2
                cy = (concept['l'][1] + concept['u'][1]) / 2
                cz = concept['u'][2] + offset
                save_ax.text(cx, cy, cz, label_text, color=concept['color'], fontsize=9)

            for role in data['roles']:
                box = create_3d_box(role['l'], role['u'], role['color'], alpha=0.2)
                save_ax.add_collection3d(box)

                label_text = label_map.get(role['name'], role['name'])
                cx = (role['l'][0] + role['u'][0]) / 2
                cy = (role['l'][1] + role['u'][1]) / 2
                cz = role['u'][2] + offset
                save_ax.text(cx, cy, cz, label_text, color=role['color'], fontsize=9)

            # Copy current axis limits and view angle
            save_ax.set_xlim(ax.get_xlim())
            save_ax.set_ylim(ax.get_ylim())
            save_ax.set_zlim(ax.get_zlim())
            save_ax.view_init(elev=ax.elev, azim=ax.azim)

            save_ax.set_xlabel(f'Dimension {state["dim_x"]}')
            save_ax.set_ylabel(f'Dimension {state["dim_y"]}')
            save_ax.set_zlabel(f'Dimension {state["dim_z"]}')
            save_ax.set_title(f'3D View: Dim {state["dim_x"]} vs Dim {state["dim_y"]} vs Dim {state["dim_z"]}')

            filename = f"Plots/{plot_file_name}_3D_dim{state['dim_x']}_dim{state['dim_y']}_dim{state['dim_z']}.svg"

        save_fig.savefig(filename, format="svg", bbox_inches='tight')
        plt.close(save_fig)
        print(f"Saved to {filename}")

    # Initial draw without auto-focus
    draw_2d(state['dim_x'], state['dim_y'], auto_focus=False)

    plt.show()

    return fig, ax


def plot_and_save_solution_weak(complexConcepts, complexRoles, label_positions, axis_limits,
                                label_map, colorMap, dimensionality, x_arr, x_index_dict,
                                plot_file_name, individuals_positions=None, individuals_bumps=None,
                                triples=None, plot_axis=False, interactive=True):
    """
    Plot box embeddings with optional interactive dimension selection.

    Args:
        interactive: If True, show interactive plot with dimension sliders and 2D/3D toggle.
                     If False, use original static plotting (dims 0 and 1).
    """
    if interactive and dimensionality > 2:
        return plot_and_save_solution_interactive(
            complexConcepts, complexRoles, label_map, colorMap, dimensionality, x_arr, x_index_dict, plot_file_name,
            individuals_positions, individuals_bumps, triples
        )
    else:
        # Original static 2D plotting code
        if individuals_positions is None:
            individuals_positions = []
        offset = 0.02
        fig, ax = plt.subplots()

        if individuals_positions != []:
            for point in individuals_positions:
                x, y = individuals_positions[point][0], individuals_positions[point][1]
                plt.scatter(x, y, c='black', s=50, zorder=5)
                plt.text(x + offset, y + offset, point, color='black')
            for triple in triples:
                i1 = triple[0]
                i2 = triple[2]
                xh, yh = individuals_positions[i1][0] + individuals_bumps[i2][0], individuals_positions[i1][1] + \
                         individuals_bumps[i2][1]
                xt, yt = individuals_positions[i2][0] + individuals_bumps[i1][0], individuals_positions[i2][1] + \
                         individuals_bumps[i1][1]
                plt.scatter(xh, yh, c='orange', s=50, zorder=5)
                plt.scatter(xt, yt, c='purple', s=50, zorder=5)
                plt.text(xh + offset, yh + offset, i1, color='orange')
                plt.text(xt + offset, yt + offset, i2, color='purple')

        all_x = []
        all_y = []

        for c in complexConcepts:
            c_name = c.get_op_name()
            l, u = getTransformedBox(c, dimensionality, x_arr, x_index_dict, True)
            l = l.value
            u = u.value

            all_x.extend([l[0], u[0]])
            all_y.extend([l[1], u[1]])

            center = getCenter(l, u)
            rect = patches.Rectangle((l[0], l[1]),
                                     u[0] - l[0],
                                     u[1] - l[1],
                                     linewidth=1, edgecolor=colorMap[c_name], facecolor='none', alpha=0.8)

            if label_positions is None:
                label_position_x = center[0]
                label_position_y = u[1] + offset * 2
            else:
                label_position_x = label_positions[c_name][0]
                label_position_y = label_positions[c_name][1]

            plt.text(label_position_x, label_position_y, label_map[c_name], color=colorMap[c_name])
            ax.add_patch(rect)

        for r in complexRoles:
            r_name = r.get_op_name()
            hl, hu, tl, tu, bl, bu = getTransformedBox(r, dimensionality, x_arr, x_index_dict, True)

            rel_dic = {'h': (hl, hu), 't': (tl, tu)}

            for pos in rel_dic:
                r_name_pos = r_name + "_" + pos
                (l, u) = rel_dic[pos]
                l = l.value
                u = u.value

                all_x.extend([l[0], u[0]])
                all_y.extend([l[1], u[1]])

                rect = patches.Rectangle((l[0], l[1]),
                                         u[0] - l[0],
                                         u[1] - l[1],
                                         linewidth=1, edgecolor=colorMap[r_name_pos], facecolor='none', alpha=0.8)

                label_position_x = label_positions[r_name_pos][0]
                label_position_y = label_positions[r_name_pos][1]

                plt.text(label_position_x, label_position_y, label_map[r_name_pos], color=colorMap[r_name_pos])
                ax.add_patch(rect)

        if plot_axis:
            plt.xlim(axis_limits[0])
            plt.ylim(axis_limits[1])
        else:
            if all_x and all_y:
                x_margin = (max(all_x) - min(all_x)) * 0.1
                y_margin = (max(all_y) - min(all_y)) * 0.1
                plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
                plt.ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        plt.savefig("Plots/" + plot_file_name + ".svg", format="svg")
        plt.show()

        return fig, ax
