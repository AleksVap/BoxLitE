S_OMEGA = 1
epsilon = 1e-2

EVALUATION_MODES = [
                   'Test',
                   'Train',
                   'Validation',
                    ]

EVALUATION_PREFIXES = {
                        'Test': '/Test_',
                        'Train': '/Train_',
                        'Validation': '/Validation_',
}

LABEL_MAP = {
    'Female': 'η(Female)',
    'Mother': 'η(Mother)',
    'Male': 'η(Male)',
    'Father': 'η(Father)',
    'exists(hasRelative)': 'η(∃hasRelative)',
    'exists(inv(hasRelative))': 'η(∃hasRelative⁻)',
    'exists(hasSibling)': 'η(∃hasSibling)',
    'exists(inv(hasSibling))': 'η(∃hasSibling⁻)',
    'exists(hasChild)': 'η(∃hasChild)',
    'exists(inv(hasChild))': 'η(∃hasChild⁻)',
    'exists(hasMother)': 'η(∃hasMother)',
    'exists(inv(hasMother))': 'η(∃hasMother⁻)',
    'exists(hasFather)': '(∃hasFather)',
    'exists(inv(hasFather))': 'η(∃hasFather⁻)',
    'exists(hasParent)': 'η(∃hasParent)',
    'exists(inv(hasParent))': 'η(∃hasParent⁻)',
    'exists(spouse)': 'η(∃spouse)',
    'exists(inv(spouse))': 'η(∃spouse⁻)',
    'exists(relative)': 'η(∃relative)',
    'exists(inv(relative))': 'η(∃relative⁻)',
    'Parent': 'η(Parent)',
    'Person': 'η(Person)',
    'exists(r)': 'η(∃r)',
    'exists(s)': 'η(∃s)',
    'exists(inv(s))': 'η(∃s⁻)',
    'exists(inv(r))': 'η(∃r⁻)',
    'neg(exists(r))': 'η(¬∃r)',
    'neg(exists(s))': 'η(¬∃s)',
    'neg(exists(inv(s)))': 'η(¬∃s⁻)',
}

CONCEPT_COLOR_MAP = {
    'Female': "slategrey",
    'Mother': "orangered",
    'Male': "olivedrab",
    'Father': "magenta",
    'exists(hasSibling)': "teal",
    'exists(inv(hasSibling))': "teal",
    'Parent': "dodgerblue",
    'Person': "goldenrod",
    'exists(hasChild)': "dodgerblue",
    'exists(inv(hasChild))': "goldenrod",
    'exists(hasRelative)': "teal",
    'exists(inv(hasRelative))': "yellow",
    'exists(hasMother)': 'orangered',
    'exists(inv(hasMother))': 'blue',
    'exists(hasFather)': 'magenta',
    'exists(inv(hasFather))': 'green',
    'exists(hasParent)': 'dodgerblue',
    'exists(inv(hasParent))': 'purple',
    'exists(spouse)': 'gray',
    'exists(inv(spouse))': 'pink',
    'exists(relative)': 'cyan',
    'exists(inv(relative))': 'brown',
    'exists(r)': 'red',
    'exists(s)': 'blue',
    'exists(inv(s))': 'green',
    'exists(inv(r))': "olivedrab",
    'neg(exists(r))': 'orange',
    'neg(exists(s))': 'teal',
    'neg(exists(inv(s)))': 'magenta',
}

AXIS_LIMITS = [(-1.25, 1.25), (-1.75, 0.75)]

def parse_kwargs(**kwargs):
    if 'dataset' in kwargs.keys():
        dataset = kwargs['dataset']
    else:
        raise ValueError("The dataset must be specified.")

    if 'model' in kwargs.keys():
        model = kwargs['model']
    else:
        raise ValueError("The SGD model name must be specified.")

    return model, dataset
