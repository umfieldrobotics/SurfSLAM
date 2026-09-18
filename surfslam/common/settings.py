"""
File: src/common/settings.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

from typing import List
import io
import os
import yaml
import numpy as np
import copy

import attridict

from common.dataset_paths import expand_tokens, load_config

class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        super().__init__(stream)

def _load_with_tokens(path: str):
    """Parse a settings yaml with ${key}/PROJECT_ROOT tokens expanded first.

    SettingsLoader resolves !include relative to stream.name, so the expanded
    text is fed through a named StringIO instead of the file handle.
    """
    with open(path, "r") as f:
        text = expand_tokens(f.read())
    stream = io.StringIO(text)
    stream.name = path
    return yaml.load(stream, SettingsLoader)

def _include(loader, node):
    fname = os.path.join(loader._root, loader.construct_scalar(node))
    return _load_with_tokens(fname)

def _project_root_constructor(loader, node):
    value = loader.construct_scalar(node)
    return os.path.join(loader._project_root, value)

SettingsLoader.add_constructor("!include", _include)
SettingsLoader.add_constructor("!project_root", _project_root_constructor)

def generate_change_list(changes):

    options = []

    # Recursively parse overrides looking for leaf elements. 
    # build options as (path_to_setting: List[str], options: List[Any])
    def _generate_options_helper(data, stack):
        if not isinstance(data, dict):
            options.append((tuple(stack), data))
            return
        
        for element in data:
            _generate_options_helper(data[element], stack + [element])
    
    _generate_options_helper(changes, [])

    return options



class Settings(attridict.AttriDict):
    """Settings wrapper that avoids AttriDict recursive __init__"""

    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Settings expects a dict")

        # Initialize dict directly (bypass AttriDict.__init__)
        dict.__init__(self)

        # Recursively convert nested dicts to AttriDict
        def convert(x):
            if isinstance(x, dict):
                return Settings(x)
            return x

        for k, v in data.items():
            self[k] = convert(v)

    @staticmethod
    def load_from_file(path: str):
        return Settings(_load_with_tokens(path))

    def augment(self, changes):

        if changes is not None:
            change_list = generate_change_list(changes)    
            
            for attr_stack, value in change_list:
                element = self
                for attr in attr_stack[:-1]:
                    element = element[attr]
                element[attr_stack[-1]] = value

    def generate_options(filename: str, overrides: str, run_all_combos: bool = False, augmentations: List[dict] = None):
        """
        @param filename: Baseline settings
        @param overrides: Path to file specifying which parameters to change, and what possible values.

        When run_all_combos is false, it will run the baselines and change one thing at a time.

        Given a settings file and overrides, computes all possible combinations of settings.

        For example, consider the baseline settings (in @p filename) are:

        mapper:
            optimizer:
                num_iterations: 10
                num_samples: 20
            num_keyframes: 20
        tracker:
            num_icp_iterations: 20
        
        
        And overrides are:

        mapper:
            optimizer:
                num_iterations: [5,10,15]
        tracker:
            num_icp_iterations: [10, 30]

        This function will return 6 sets of settings, with all the combinations of settings specified in the overrrides,
        and everything else as specified in the baseline.

        @returns a list of settings with all combinations of options in overrides, and everything else left at baseline
        """

        baseline = Settings.load_from_file(filename)

        if augmentations is not None:
            for changes in augmentations:
                if changes is not None:
                    baseline.augment(changes)

        overrides_datas = load_config(overrides)

        if not isinstance(overrides_datas, list):
            overrides_datas = [overrides_datas]

        all_settings_options, all_settings_descriptions = [], []

        for overrides_data in overrides_datas:

            if overrides_data is None:
                continue
            
            options = generate_change_list(overrides_data)

            for idx, (key, values) in enumerate(options):
                if not isinstance(values, list):
                    options[idx] = (key, [values])

            if run_all_combos:
                # How many choices are there for each override
                option_counts = [len(o[1]) for o in options]

                # Build combinations of overrides, as indices in the array
                all_index_options = tuple(np.arange(o) for o in option_counts)
                all_idx_combos = np.array(np.meshgrid(*all_index_options)).T.reshape(-1,len(all_index_options))

                attr_stacks = [o[0] for o in options]

                # Make a copy of the settings for each combo of settings 
                settings_options = []
                settings_descriptions = []
                for idx_combo in all_idx_combos:
                    settings_copy = copy.deepcopy(baseline)
                    settings_description = ""
                    for attr_idx, (option_idx, attr_stack) in enumerate(zip(idx_combo, attr_stacks)):
                        element = settings_copy
                        for attr in attr_stack[:-1]:
                            element = element[attr]
                        attr_val = options[attr_idx][1][option_idx]
                        element[attr_stack[-1]] = attr_val

                        attr_path = ".".join(attr_stack)
                        settings_description += f"{attr_path}={attr_val}\n"
                    settings_options.append(settings_copy)
                    settings_descriptions.append(settings_description)

                all_settings_options += settings_options
                all_settings_descriptions += settings_descriptions
            else:
                settings_options = []
                settings_descriptions = []
                for attr_stack, values in options:
                    
                    if len(values) > 0 and isinstance(values[0], list):
                        values = [values]

                    for value in values:
                        settings_copy = copy.deepcopy(baseline)
                        settings_description = ""

                        element = settings_copy
                        for attr in attr_stack[:-1]:
                            element = element[attr]
                        element[attr_stack[-1]] = value

                        attr_path = ".".join(attr_stack)
                        settings_description = f"{attr_path}={value}"
                        
                        settings_options.append(settings_copy)
                        settings_descriptions.append(settings_description)

                all_settings_options += settings_options
                all_settings_descriptions += settings_descriptions

        if len(all_settings_options) == 0:
            all_settings_options = [baseline]
            all_settings_descriptions = [""]
        return all_settings_options, all_settings_descriptions