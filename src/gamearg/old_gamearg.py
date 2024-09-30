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


def get_graphviz_schema_and_facts_prep(input_file, keyword, reverse=False):
    """Read graphviz settings and prepare facts."""
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, input_file)
    with open(path_to_json, 'r') as file:
        schema = json.load(file)

    status_1 = schema[keyword]["status"]["status_1"]
    status_2 = schema[keyword]["status"]["status_2"]
    status_3 = schema[keyword]["status"]["status_3"]

    facts_prep = f"e(X,Y):- attacks(Y,X)." if reverse else f"e(X,Y):- attacks(X,Y)."

    return status_1, status_2, status_3, facts_prep


def get_status(node, pw, nodes_status):
    """Retrieve status of a node in a given possible world."""
    for status, nodes in nodes_status[pw].items():
        if node in nodes:
            return status
    return None


def add_multiple_pws_to_df(node_wfs_df, nodes_status):
    """Update a DataFrame with multiple possible worlds statuses."""
    for pw_key in nodes_status:
        node_wfs_df[pw_key] = node_wfs_df["node"].apply(
            lambda x: get_status(x, pw_key, nodes_status)
        )
    return node_wfs_df


def node_wfs_cal(input_file, keyword, reverse=False):
    """function to calculate the state of the game."""
    status_1, status_2, status_3, facts_prep = get_graphviz_schema_and_facts_prep(
        "graphviz_settings.json", keyword, reverse
    )
    statelog = f"""
    #const state_max = 100. 

    {facts_prep}

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
    len({status_2},X,L)   :- fr(S,X), L = 2*S.   % Two plies = one e:  len = 0,2,4, ..
    len({status_1},X,L)    :- fg(S,X), L = 2*S-1. % Green is 1 ply behind: len = 1,3,5, ..
    len({status_3},X,"∞") :- pos(X), not len({status_2},X,_), not len({status_1},X,_). % Gap = draws

    #show final/1.
    #show fr/2.
    #show fg/2.
    #show len/3.
    """
    ctl = clingo.Control()
    if "files/" not in input_file:
        ctl.add("base", [], input_file)
    else:
        ctl.load(input_file)
    
    ctl.add("base", [], statelog)
    ctl.ground([("base", [])])
    models = []
    ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)))
    atoms = models[0]
    nodes = []
    for atom in atoms:
        if atom.name == "fr" and atom.arguments[0] == 0:
            nodes.append((str(atom.arguments[1]), 0, status_2))
        if atom.name == "len":
            if str(atom.arguments[0]) == status_3:
                nodes.append((str(atom.arguments[1]), "∞", str(atom.arguments[0])))
            else:
                nodes.append(
                    (
                        str(atom.arguments[1]),
                        str(atom.arguments[2]),
                        str(atom.arguments[0]),
                    )
                )
    node_wfs_df = pd.DataFrame(nodes, columns=["node", "state_id", "wfs"])
    node_wfs_df = node_wfs_df.sort_values(by="state_id")
    return node_wfs_df


def node_stb_cal(input_file, keyword, reverse=False):
    status_1, status_2, status_3, facts_prep = get_graphviz_schema_and_facts_prep(
        "graphviz_settings.json", keyword, reverse
    )
    stablelog = f"""
    {facts_prep}
    pos(X) :- e(X,_).
    pos(X) :- e(_,X).

    {status_1}(X) :- e(X,Y), {status_2}(Y).
    {status_2}(X) :- pos(X), not {status_1}(X).

    #show {status_1}/1.
    #show {status_2}/1.
    """
    ctl = clingo.Control()
    ctl.configuration.solve.models = "0"
    ctl.load(input_file)
    ctl.add("base", [], stablelog)
    ctl.ground([("base", [])])
    stb_output = []
    ctl.solve(on_model=lambda m: stb_output.append(m.symbols(shown=True)))
    nodes_status = {}
    pw_id = 1
    for model in stb_output:
        status_2_ls = []
        status_1_ls = []
        nodes_status[f"pw_{pw_id}"] = {}
        for atom in model:
            if atom.name == status_2:
                status_2_ls.append(str(atom.arguments[0]))
                nodes_status[f"pw_{pw_id}"][status_2] = status_2_ls
            elif atom.name == status_1:
                status_1_ls.append(str(atom.arguments[0]))
                nodes_status[f"pw_{pw_id}"][status_1] = status_1_ls
        pw_id = pw_id + 1
    node_wfs_df = node_wfs_cal(input_file, keyword, reverse)
    node_pws_df = add_multiple_pws_to_df(node_wfs_df, nodes_status)
    df_wfs_stb = node_pws_df.sort_values(
        by=["state_id", "wfs"], ascending=[True, True]
    ).reset_index(drop=True)
    wfs_stb_pws = list(nodes_status.keys())
    wfs_stb_pws.append("wfs")
    return wfs_stb_pws, df_wfs_stb


def read_edges_from_file(input_file, keyword, reverse):
    try:
        with open(input_file, "r") as file:
            edges = []
            prg = []
            ast.parse_files([input_file], lambda x: prg.append(x))
            for line in prg[1:]:
                try:
                    start_node = line.head.atom.symbol.arguments[0].values()[1].name
                    end_node = line.head.atom.symbol.arguments[1].values()[1].name
                except (AttributeError,RuntimeError) as e:
                    start_node = str(line.head.atom.symbol.arguments[0].values()[1])
                    end_node = str(line.head.atom.symbol.arguments[1].values()[1])
                if keyword == "game":
                    direction = "forward"
                else:
                    direction = "back"
                if reverse:
                    edges.append((end_node, start_node, direction))
                else:
                    edges.append((start_node, end_node, direction))
    except FileNotFoundError:
        raise FileNotFoundError(f"{input_file} not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")
    edge_df = pd.DataFrame(edges, columns=["source", "target", "direction"])
    return edge_df


# Color Nodes
def create_label(status_1, status_2, row):
    try:
        row["node"] = int(row["node"])
        row["node"] = "p" + str(row["node"])
    except ValueError:
        pass
    if '"' in row["node"]:
        row["node"] = row["node"].replace('"', "")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        schema = json.load(file)
    isExistential=schema["show_existential_quantification"]
    if isExistential==True:
        exist_symbol = "∃"
        all_symbol = "∀"
    else:
        exist_symbol = ""
        all_symbol = ""
    if row["wfs"] == status_2:  # assuming 'status' key should be 'wfs'
        return f"{all_symbol} {row['node']}.{row['state_id']}"
    elif row["wfs"] == status_1:  # assuming 'status' key should be 'wfs'
        return f"{exist_symbol} {row['node']}.{row['state_id']}"
    else:
        return f"{row['node']}.{row['state_id']}"


def get_color(status, keyword):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        color_settings = json.load(file)
    return color_settings[keyword]["node_color"].get(status, "default_color")


def get_color_for_row(row, pw, keyword, status_3):
    if row["wfs"] == status_3:
        return get_color(f"{status_3}_{row[pw]}", keyword)
    else:
        return get_color(row[pw], keyword)


def get_node_properties(input_file, keyword, reverse=False):
    status_1, status_2, status_3, facts_prep = get_graphviz_schema_and_facts_prep(
        "graphviz_settings.json", keyword, reverse
    )
    wfs_stb_pws, df_wfs_stb = node_stb_cal(input_file, keyword, reverse)

    colored_node_df = df_wfs_stb.copy()

    for pw in wfs_stb_pws:
        if "pw" in pw:
            colored_node_df[pw + "_color"] = colored_node_df.apply(
                lambda row: get_color_for_row(row, pw, keyword, status_3), axis=1
            )
        else:
            colored_node_df[pw + "_color"] = colored_node_df.apply(
                lambda row: get_color(row[pw], keyword), axis=1
            )

    colored_node_df["label"] = colored_node_df.apply(
        lambda row: create_label(status_1, status_2, row), axis=1
    )

    return colored_node_df.sort_values(
        by=["state_id", "wfs"], ascending=[True, True]
    ).reset_index(drop=True)


# Color Edges
def get_edge_color(source_status, target_status, schema, keyword="game"):
    """Retrieve edge properties based on source and target statuses."""
    for edge_setting in schema[keyword]["edge_color"]:
        if (
            edge_setting["source"] == source_status
            and edge_setting["target"] == target_status
        ):
            return (
                edge_setting["color"],
                edge_setting["style"],
                edge_setting.get("show_label", True),
            )
    return "default_color", "default_style", True


def can_be_number(s):
    """Check if a string can be converted to a number."""
    try:
        float(s)  # try to convert to a float
        return True
    except ValueError:
        return False


def get_edge_properties(input_file, keyword="game", reverse=False):
    """Process edge data to add color, style, and label based on node statuses."""

    wfs_stb_pws, df_wfs_stb = node_stb_cal(input_file, keyword, reverse)
    edge_df = read_edges_from_file(input_file, keyword, reverse)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        schema = json.load(file)

    merged_df = edge_df.merge(
        df_wfs_stb,
        left_on="source",
        right_on="node",
        how="left",
        suffixes=("", "_source"),
    )
    merged_df = merged_df.merge(
        df_wfs_stb,
        left_on="target",
        right_on="node",
        how="left",
        suffixes=("_source", "_target"),
    )

    # For each status, apply get_edge_properties function and create the color columns
    for status in wfs_stb_pws:
        merged_df[f"{status}_edge_color"], merged_df[f"{status}_edge_style"], _ = zip(
            *merged_df.apply(
                lambda row: get_edge_color(
                    row[f"{status}_source"], row[f"{status}_target"], schema, keyword
                ),
                axis=1,
            )
        )
    # print(merged_df)
    merged_df["edge_label"] = merged_df.apply(
        lambda row: ""
        if row["wfs_edge_color"] == "black"
        else str(int(row["state_id_target"]) + 1)
        if can_be_number(row["state_id_target"])
        else row["state_id_target"],
        axis=1,
    )
    # Select the required columns for the final DataFrame
    colored_edge_df = merged_df[
        ["source", "target", "direction", "edge_label"]
        + [f"{status}_edge_color" for status in wfs_stb_pws]
        + [f"{status}_edge_style" for status in wfs_stb_pws]
    ]
    return colored_edge_df


# Generate Graphviz
def group_edges(input_list):
    edge_groups = defaultdict(list)
    result = []
    brace_count = 0  # Count of open braces to identify the last closing brace.

    # Parsing the input list.
    for index, line in enumerate(input_list):
        brace_count += line.count("{") - line.count("}")  # Update brace_count

        # If brace_count is zero after counting braces in the line, it means
        # we are at the last closing brace.
        if "}" in line and brace_count == 0:
            continue  # Don't add the closing } yet, we will add it at the end.

        # Extracting edges and their properties
        elif "->" in line:
            edge_prop_index = line.find("[")
            if edge_prop_index != -1:
                edge_str = line[:edge_prop_index].rstrip()
                prop_str = line[edge_prop_index:].rstrip().rstrip(";")
                edge_groups[prop_str].append(edge_str.rstrip(";").rstrip())

        # Preserving node, subgraph information, and any other lines.
        else:
            result.append(line)  # Preserve other lines as they are.

    # Sort edge_groups so that items with `constraint=false` are at the end.
    sorted_edge_props = sorted(
        edge_groups.keys(), key=lambda prop: "constraint=false" in prop
    )

    # Grouping edges with the same properties
    for props in sorted_edge_props:
        edges = edge_groups[props]
        result.append(f"    edge {props}\n")
        result.extend([f"       {edge} \n" for edge in edges])

    return result


def generate_clean_dot_string(
    colored_node_df,
    colored_edge_df,
    model,
    keyword,
    input_file,
    reverse_str,
    edge_color_col="wfs_edge_color",
    edge_style_col="wfs_edge_style",
):
    node_color_col = model + "_color"

    # Group nodes by color for the specific semantics or possible world
    grouped_nodes_by_color = (
        colored_node_df.groupby(node_color_col)
        .apply(lambda x: x[["node", "label"]].values.tolist())
        .to_dict()
    )
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        schema = json.load(file)
    labeldistance=schema["labeldistance"]
    # Initialize DOT string
    dot_string = "digraph {\n"
    dot_string += '    rankdir="TB"\n'
    dot_string += '    node [style=filled fontname="Palatino" fontsize=14]\n'

    # Adding nodes grouped by color
    for color, nodes in grouped_nodes_by_color.items():
        dot_string += f'    node [fillcolor="{color}"]\n'
        for node, label in nodes:
            dot_string += f'    {node} [label="{label}"]\n'
    
    dot_string += f'    edge[labeldistance={labeldistance} fontsize=12]\n'
    # Construct edge strings
    edge_string_ls = []
    for _, row in colored_edge_df.iterrows():
        edge_color = (
            f'color="{row[edge_color_col]}:{row[edge_color_col]}"'
            if row[edge_color_col] != "black" and row["wfs_edge_color"] == "black"
            else f'color="{row[edge_color_col]}"'
        )
        edge_style = "dotted" if row[edge_color_col] == "black" else row[edge_style_col]
        edge_label = row["edge_label"]
        edge_dir = row["direction"]
        constraint = " constraint=false" if row["wfs_edge_color"] == "black" else ""
        edge = f'   {row["source"]} -> {row["target"]} [{edge_color} style="{edge_style}" dir="{edge_dir}" taillabel="{edge_label}"{constraint}]\n'
        edge_string_ls.append(edge)

    # Assuming 'group_edges' is a function that groups edge strings in some way
    grouped_edge_string = (
        group_edges(edge_string_ls) if "group_edges" in globals() else edge_string_ls
    )
    for edge_string in grouped_edge_string:
        dot_string += edge_string

    # Rank nodes with the same state_id together if they are numeric
    numeric_state_ids = pd.to_numeric(
        colored_node_df["state_id"], errors="coerce"
    ).dropna()
    min_state_id, max_state_id = numeric_state_ids.min(), numeric_state_ids.max()
    if np.isnan(min_state_id) or np.isnan(max_state_id):
        dot_string += " "
    else:
    # Group by state_id and construct rank strings
        for state_id, group in colored_node_df.groupby("state_id"):
            if state_id == str(int(min_state_id)):
                rank_label = "max"
            elif state_id == str(int(max_state_id)):
                rank_label = "min"
            else:
                continue
            nodes_same_rank = " ".join(node for node in group["node"])
            dot_string += f"    {{rank = {rank_label} {nodes_same_rank}}}\n"

    # Close the DOT string
    dot_string += "}\n"
    if "files" in input_file:
        graph_folder=input_file.split(".")[0].split("/")[1]
    else:
        graph_folder=input_file.split(".")[0]
    folder_name = "imgs"+"/"+graph_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(f"imgs/{graph_folder}/clean_{reverse_str}_{keyword}_{model}.dot", "w") as file:
        file.write(dot_string)


def generate_dot_string(
    colored_node_df,
    colored_edge_df,
    model,
    keyword,
    input_file,
    reverse_str,
    edge_color_col="wfs_edge_color",
    edge_style_col="wfs_edge_style",
):
    node_color_col = model + "_color"
    dot_string = "digraph {\n"
    dot_string += "    // Node defaults can be set here if needed\n"

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        schema = json.load(file)
    labeldistance=schema["labeldistance"]
    # Adding node information
    for index, row in colored_node_df.iterrows():
        node = f'    {row["node"]} [style="filled" fillcolor="{row[node_color_col]}" label="{row["label"]}" fontsize=14]\n'
        dot_string += node

    dot_string += f'    edge[labeldistance={labeldistance} fontsize=12]\n'
    # Adding edge information
    for index, row in colored_edge_df.iterrows():
        constraint = "constraint=false" if row["wfs_edge_color"] == "black" else ""
        color = (
            f'color="{row[edge_color_col]}:{row[edge_color_col]}"'
            if row[edge_color_col] != "black" and row["wfs_edge_color"] == "black"
            else f'color="{row[edge_color_col]}"'
        )

        edge_style = "dotted" if row[edge_color_col] == "black" else row[edge_style_col]

        edge = f'{row["source"]} -> {row["target"]} [{color} style="{edge_style}" dir="{row["direction"]}" taillabel="{row["edge_label"]}" {constraint}]\n'
        dot_string += "    "+edge

    numeric_state_ids = pd.to_numeric(
        colored_node_df["state_id"], errors="coerce"
    ).dropna()
    min_state_id, max_state_id = numeric_state_ids.min(), numeric_state_ids.max()
    if np.isnan(min_state_id) or np.isnan(max_state_id):
        dot_string += " "
    else:
        for state_id, group in colored_node_df.groupby("state_id"):
            if state_id == str(int(min_state_id)):
                rank_label = "max"
            elif state_id == str(int(max_state_id)):
                rank_label = "min"
            else:
                continue
            nodes_same_rank = " ".join(node for node in group["node"])
            dot_string += f"    {{rank = {rank_label} {nodes_same_rank}}}\n"

    dot_string += "}"

    graph_folder=input_file.split(".")[0].split("/")[1]
    folder_name = "imgs"+"/"+graph_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(f"imgs/{graph_folder}/unfactored_{reverse_str}_{keyword}_{model}.dot", "w") as file:
        file.write(dot_string)

def generate_plain_dot_string(
    colored_node_df,
    colored_edge_df,
    model,
    keyword,
    input_file,
    reverse_str
):
    dot_string = "digraph {\n"
    dot_string += "    // Node defaults can be set here if needed\n"

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path_to_json = os.path.join(dir_path, "graphviz_settings.json")
    with open(path_to_json, "r") as file:
        schema = json.load(file)
    labeldistance=schema["labeldistance"]
    # Adding node information
    for index, row in colored_node_df.iterrows():
        node = f'    {row["node"]} [fontsize=14]\n'
        dot_string += node

    dot_string += f'    edge[labeldistance={labeldistance} fontsize=12]\n'
    # Adding edge information
    for index, row in colored_edge_df.iterrows():
        edge = f'{row["source"]} -> {row["target"]} [dir="{row["direction"]}"]\n'
        dot_string += "    "+edge
    dot_string += "}"
    graph_folder=input_file.split(".")[0].split("/")[1]
    folder_name = "imgs"+"/"+graph_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(f"imgs/{graph_folder}/plain_{reverse_str}_{keyword}.dot", "w") as file:
        file.write(dot_string)


def render_dot_to_png(dot_file_path, output_file_path):
    try:
        # Run the command to convert the DOT file to a PNG file
        subprocess.run(
            ["dot", "-Tpng", dot_file_path, "-o", output_file_path], check=True
        )

    except FileNotFoundError:
        print(f"File {dot_file_path} does not exist.")
    except subprocess.CalledProcessError:
        print(f"Error occurred while converting {dot_file_path} to PNG.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_graphviz(input_file, keyword, reverse=False):
    """Generate Graphviz dot string for the input file."""
    clear_imgs_folder(input_file)
    colored_node_df = get_node_properties(input_file, keyword, reverse)
    colored_edge_df = get_edge_properties(input_file, keyword, reverse)
    # print(colored_edge_df)


    if reverse:
        reverse_str = "backward"
    else:
        reverse_str = "forward"


    wfs_stb_pws, df_wfs_stb = node_stb_cal(input_file, keyword, reverse)
    for pw in wfs_stb_pws:
        generate_plain_dot_string(
            colored_node_df,
            colored_edge_df,
            pw,
            keyword,
            input_file,
            reverse_str)

        generate_dot_string(
            colored_node_df,
            colored_edge_df,
            pw,
            keyword,
            input_file,
            reverse_str,
            edge_color_col=pw + "_edge_color",
        )
        generate_clean_dot_string(
            colored_node_df,
            colored_edge_df,
            pw,
            keyword,
            input_file,
            reverse_str,
            edge_color_col=pw + "_edge_color",
        )

    # Generate PNG files
    graph_folder=input_file.split(".")[0].split("/")[1]
    render_dot_to_png(
            f"imgs/{graph_folder}/plain_{reverse_str}_{keyword}.dot", f"imgs/{graph_folder}/plain_{reverse_str}_{keyword}.png"
        )
    for pw in wfs_stb_pws:
        render_dot_to_png(
            f"imgs/{graph_folder}/unfactored_{reverse_str}_{keyword}_{pw}.dot", f"imgs/{graph_folder}/unfactored_{reverse_str}_{keyword}_{pw}.png"
        )
        render_dot_to_png(
            f"imgs/{graph_folder}/clean_{reverse_str}_{keyword}_{pw}.dot", f"imgs/{graph_folder}/clean_{reverse_str}_{keyword}_{pw}.png"
        )


def clear_imgs_folder(input_file):
    graph_folder=input_file.split(".")[0].split("/")[1]
    folder_path = "imgs/"+graph_folder

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Remove all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectories if any
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def display_images_in_rows(input_file, file_prefix, images_per_row=2, image_width="auto"):
    """
    Display images in rows from a specified folder.

    Args:
    folder_path (str): Path to the folder containing images.
    file_prefix (str): Prefix of the image files to display.
    images_per_row (int): Number of images to display per row.
    image_width (str): Width of each image (CSS value).
    """
    # Gather all relevant .png files
    folder_path= "imgs/"+input_file.split(".")[0].split("/")[1]
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith(file_prefix) and f.endswith(".png")
    ]
    image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split("_")[4].split(".")[0]))
    if len(image_files) == 0:
        print(f"Notice: this move(attack) graph doesn't have stable models(extensions).")
    # Start creating the HTML string
    html_str = ""

    for i, img_file in enumerate(image_files):
        # Extract pw_id from the file name
        pw_id = os.path.basename(img_file).split("_")[4].split(".")[0]

        if i % images_per_row == 0:
            html_str += '<div style="text-align: center;">'  # Start a new row

        html_str += (
            f'<div style="display: inline-block; width: {image_width}; margin: 60px;">'
        )
        html_str += f'<img src="{img_file}" style="width: 100%; height: auto;" alt="PW {pw_id}" />'
        html_str += f"<div>PW {pw_id}</div></div>"  # Adding title

        if i % images_per_row == images_per_row - 1 or i == len(image_files) - 1:
            html_str += "</div>"  # End the row

    # Display the HTML
    display(HTML(html_str))

def show_plain(input_file, keyword="arg", reverse=True):
    generate_graphviz(input_file, keyword, reverse)
    # Extracting the folder name from the input file
    graph_folder = input_file.split(".")[0].split("/")[1]
    if reverse:
        reverse_str = "backward"
    else:
        reverse_str = "forward"
    image_file=f"imgs/{graph_folder}/plain_{reverse_str}_{keyword}.png"
    # Displaying the image
    return Image(image_file)

def show_wfs(input_file, keyword="arg", reverse=True, gvz_version="unfactored"):
    # Generate the Graphviz graph
    generate_graphviz(input_file, keyword, reverse)

    # Extracting the folder name from the input file
    if "files" in input_file:
        graph_folder=input_file.split(".")[0].split("/")[1]
    else:
        graph_folder=input_file.split(".")[0]
    
    if reverse:
        reverse_str = "backward"
    else:
        reverse_str = "forward"
    # Constructing the image file path
    image_file = f"imgs/{graph_folder}/{gvz_version}_{reverse_str}_{keyword}_wfs.png"

    # Displaying the image
    return Image(image_file)

def show_stb(input_file, keyword="arg", reverse=True, gvz_version="unfactored"):
    generate_graphviz(input_file, keyword, reverse)
    if reverse:
        reverse_str = "backward"
    else:
        reverse_str = "forward"
    image_prefix = f"{gvz_version}_{reverse_str}_{keyword}_pw"
    display_images_in_rows(input_file, image_prefix)