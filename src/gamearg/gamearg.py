# import statements
import clingo
import pandas as pd
import json
import clingo.ast as ast
from collections import defaultdict
import os
import subprocess
import shutil
from IPython.display import HTML, display, Image
import numpy as np


class GameArgInput:
    def __init__(self, input_file, keyword, reverse=False, target_pred=None):
        self.input_file = input_file
        self.keyword = keyword
        self.reverse = reverse
        self.schema = self._load_schema()
        if target_pred:
            self.pred_name = target_pred
        else:
            self.pred_name = self._get_pred_name()
        (
            self.status_1,
            self.status_2,
            self.status_3,
        ) = self._extract_status()
        self.facts_prep = self._facts_prep()
        self.edge_df = self._extract_edges()

    def _load_schema(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path_to_json = os.path.join(dir_path, "graphviz_settings.json")
        with open(path_to_json, "r") as file:
            schema = json.load(file)
        return schema

    def _get_pred_name(self):
        ctl = clingo.Control()
        ctl.load(self.input_file)
        ctl.ground([("base", [])])
        stb_output = []
        ctl.solve(on_model=lambda m: stb_output.append(m.symbols(shown=True)))
        pred_name = set()
        for model in stb_output:
            for atom in model:
                pred_name.add(atom.name)
        if len(pred_name) == 1:
            return pred_name.pop()
        else:
            raise Exception(
                "Expected exactly one unique predicate in the input file, found {}: {}".format(
                    len(pred_name), pred_name
                )
            )

    def _extract_status(self):
        status_1 = self.schema[self.keyword]["status"]["status_1"]
        status_2 = self.schema[self.keyword]["status"]["status_2"]
        status_3 = self.schema[self.keyword]["status"]["status_3"]
        return status_1, status_2, status_3

    def _facts_prep(self):
        facts_prep = (
            f"e(X,Y):- {self.pred_name}(Y,X)."
            if self.reverse
            else f"e(X,Y):- {self.pred_name}(X,Y)."
        )
        return facts_prep

    def _extract_edges(self):
        try:
            edges = []
            ast_program = []
            ast.parse_files([self.input_file], lambda x: ast_program.append(x))
            for line in ast_program[1:]:
                try:
                    start_node = line.head.atom.symbol.arguments[0].values()[1].name
                    end_node = line.head.atom.symbol.arguments[1].values()[1].name
                except (AttributeError, RuntimeError):
                    start_node = str(line.head.atom.symbol.arguments[0].values()[1])
                    end_node = str(line.head.atom.symbol.arguments[1].values()[1])
                if self.keyword == "game":
                    direction = "forward"
                else:
                    direction = "back"
                if self.reverse:
                    edges.append((end_node, start_node, direction))
                else:
                    edges.append((start_node, end_node, direction))
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.input_file} not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred: {str(e)}")
        edge_df = pd.DataFrame(edges, columns=["source", "target", "direction"])
        return edge_df


class WfsModel(GameArgInput):
    def __init__(self, *args, **kwargs):
        # Case 1: If the first argument is an instance of GameArgInput
        if args and isinstance(args[0], GameArgInput):
            game_arg_input = args[0]
            super().__init__(game_arg_input.input_file, game_arg_input.keyword, game_arg_input.reverse)
        # Case 2: If separate parameters are provided
        elif 'input_file' in kwargs and 'keyword' in kwargs:
            super().__init__(kwargs['input_file'], kwargs['keyword'], kwargs.get('reverse', False))
        else:
            raise ValueError("Invalid arguments passed to WfsModel")
        
        self.wfs_node = self.wfs_cal()

    def wfs_cal(self):
        statelog = f"""
        #const state_max = 100. 

        {self.facts_prep}

        pos(X) :- e(X,_).
        pos(X) :- e(_,X).

        % win-e rule in "doubled form" (2 rounds for 1, simplifying queries & termination)
        win_o(S, X) :- e(X,Y), not win_u(S,Y), next(S,_).  % (1) 
        win_u(S1,X) :- e(X,Y), not win_o(S,Y), next(S,S1). % (2)

        % First-Green: when was a win_u first derived?
        fg(S1,X) :- next(S,S1), not win_u(S,X), win_u(S1,X).  % (3)

        % First-Red: when did a loss first drop from win_o?
        fr(0,X)  :- pos(X), not win_o(0,X). 
        fr(S1,X) :- next(S,S1), not final(S), win_o(S,X), not win_o(S1,X). % (4) 

        % (5) Generating new states, as long as necessary
        next(0,1). 
        next(S,S1) :- fg(S,X), S1 = S+1, S < state_max.                       % (5) 
            
        % Using clingo's "_" semantics to obtain final state (second last)
        final(S) :- next(S,S1), not next(S1,_). % not \exists _ ..          % (6)

        % (7,8,9) Solutions (position values) calculation with length
        len({self.status_2},X,L)   :- fr(S,X), L = 2*S.   % Two plies = one e:  len = 0,2,4, ..
        len({self.status_1},X,L)    :- fg(S,X), L = 2*S-1. % Green is 1 ply behind: len = 1,3,5, ..
        len({self.status_3},X,"∞") :- pos(X), not len({self.status_2},X,_), not len({self.status_1},X,_). % Gap = draws

        #show final/1.
        #show fr/2.
        #show fg/2.
        #show len/3.
        """
        ctl = clingo.Control()
        ctl.load(self.input_file)
        ctl.add("base", [], statelog)
        ctl.ground([("base", [])])
        models = []
        ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)))
        atoms = models[0]
        nodes = []
        for atom in atoms:
            if atom.name == "fr" and atom.arguments[0] == 0:
                nodes.append((str(atom.arguments[1]), 0, self.status_2))
            if atom.name == "len":
                if str(atom.arguments[0]) == self.status_3:
                    nodes.append((str(atom.arguments[1]), "∞", str(atom.arguments[0])))
                else:
                    nodes.append(
                        (
                            str(atom.arguments[1]),
                            str(atom.arguments[2]),
                            str(atom.arguments[0]),
                        )
                    )
        wfs_node = pd.DataFrame(nodes, columns=["node", "state_id", "wfs"])
        wfs_node = wfs_node.sort_values(by="state_id")
        return wfs_node


class StbModel(GameArgInput):
    def __init__(self, *args, **kwargs):
        # Case 1: If the first argument is an instance of GameArgInput
        if args and isinstance(args[0], GameArgInput):
            game_arg_input = args[0]
            super().__init__(game_arg_input.input_file, game_arg_input.keyword, game_arg_input.reverse)
        # Case 2: If separate parameters are provided
        elif 'input_file' in kwargs and 'keyword' in kwargs:
            super().__init__(kwargs['input_file'], kwargs['keyword'], kwargs.get('reverse', False))
        else:
            raise ValueError("Invalid arguments passed to WfsModel")
        
        self.stb_node = self.stb_cal()

    def get_status(self, node, pw, nodes_status):
        for status, nodes in nodes_status[pw].items():
            if node in nodes:
                return status
        return None

    def stb_cal(self):
        stable_log = f"""
        {self.facts_prep}
        pos(X) :- e(X,_).
        pos(X) :- e(_,X).

        {self.status_1}(X) :- e(X,Y), {self.status_2}(Y).
        {self.status_2}(X) :- pos(X), not {self.status_1}(X).

        #show {self.status_1}/1.
        #show {self.status_2}/1.
        """
        ctl = clingo.Control()
        ctl.configuration.solve.models = "0"
        ctl.load(self.input_file)
        ctl.add("base", [], stable_log)
        ctl.ground([("base", [])])
        stb_output = []
        ctl.solve(on_model=lambda m: stb_output.append(m.symbols(shown=True)))
        stb_pws = {}
        nodes = []
        pw_id = 1
        for model in stb_output:
            status_2_ls = []
            status_1_ls = []
            stb_pws[f"pw_{pw_id}"] = {}
            for atom in model:
                nodes.append(str(atom.arguments[0]))
                if atom.name == self.status_2:
                    status_2_ls.append(str(atom.arguments[0]))
                    stb_pws[f"pw_{pw_id}"][self.status_2] = status_2_ls
                elif atom.name == self.status_1:
                    status_1_ls.append(str(atom.arguments[0]))
                    stb_pws[f"pw_{pw_id}"][self.status_1] = status_1_ls
            pw_id = pw_id + 1

        stb_node = pd.DataFrame({"node": list(set(nodes))})
        for pw_key in stb_pws:
            stb_node[pw_key] = stb_node["node"].apply(lambda x: self.get_status(x, pw_key, stb_pws))
        return stb_node
