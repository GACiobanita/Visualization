import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from implementation.data_adjustments import DataAdjustment
import numpy as np

DATA_ADJUSTER = DataAdjustment()


class DendrogramGenerator:

    @staticmethod
    # coordinates needs to be a zip of dendrogram 'icoord' and 'dcoord' link positions
    def get_top_level_nodes(coordinates):
        max_x = 0
        text_y_coords = []
        for y_coords, x_coords in coordinates:
            for y in x_coords:
                if y > max_x:
                    max_x = y
                    text_y_coords = []
            for i in range(0, len(y_coords) - 1):
                if (x_coords[i] == max_x or x_coords[i + 1] == max_x) and (
                        x_coords[i] > x_coords[i + 1] or x_coords[i] < x_coords[i + 1]):
                    text_y_coords.append(y_coords[i])
        return max_x, text_y_coords

    @staticmethod
    def construct_topics_from_file(csv_data):
        file_topics = csv_data['Main Topic'].unique().tolist()
        return file_topics

    @staticmethod
    def construct_topic_vocabulary_from_file(csv_data, topics):
        topics_dicts = {}
        word_columns = list(csv_data.columns[1:-1])
        for topic in topics:
            topics_dicts[topic] = {}
        for index, row in csv_data.iterrows():
            for word_column in word_columns:
                topics_dicts[row['Main Topic']][row[word_column]] = topics_dicts.get(row['Main Topic'], {}).get(
                    row[word_column], 0) + 1
        return topics_dicts

    @staticmethod
    def construct_tree_structure(topic_dicts):
        count = 0
        tree_structure = {'Topics': []}
        for topic, word_list in topic_dicts.items():
            tree_structure['Topics'].append(topic)
            topic_keywords = []
            for word in word_list:
                if word not in tree_structure.keys():
                    topic_keywords.append(word)
                else:
                    topic_keywords.append(word + str(count))
                    count += 1
            tree_structure[topic] = topic_keywords
            for word in topic_keywords:
                tree_structure[word] = []
        return tree_structure

    @staticmethod
    def construct_linkage_matrix(tree_structure):
        graph = nx.DiGraph(tree_structure)
        nodes = graph.nodes()
        leaves = set(n for n in nodes if graph.out_degree(n) == 0)
        inner_nodes = [n for n in nodes if graph.out_degree(n) > 0]

        # Compute the size of each subtree
        subtree = dict((n, [n]) for n in leaves)
        for u in inner_nodes:
            children = set()
            node_list = list(tree_structure[u])
            while len(node_list) > 0:
                v = node_list.pop(0)
                children.add(v)
                node_list += tree_structure[v]

            subtree[u] = sorted(children & leaves)

        inner_nodes.sort(key=lambda n: len(subtree[n]))  # <-- order inner nodes ascending by subtree size, root is last

        # Construct the linkage matrix
        leaves = sorted(leaves)
        index = dict((tuple([n]), i) for i, n in enumerate(leaves))
        linkage_matrix = []
        k = len(leaves)
        for i, n in enumerate(inner_nodes):
            children = tree_structure[n]
            x = children[0]
            for y in children[1:]:
                z = tuple(sorted(subtree[x] + subtree[y]))
                i, j = index[tuple(subtree[x])], index[tuple(subtree[y])]
                linkage_matrix.append(
                    [i, j, float(len(subtree[n])), len(z)])  # <-- float is required by the dendrogram function
                index[z] = k
                subtree[z] = list(z)
                x = z
                k += 1

        for i in range(0, len(leaves)):
            leaves[i] = DATA_ADJUSTER.remove_numbers_from_string(leaves[i])

        return linkage_matrix, leaves

    def construct_dendrogram(self, linkage_matrix, leaf_names, topics):
        fig = plt.figure(figsize=(9, 15), dpi=125)

        ax = fig.add_subplot(1, 1, 1)
        # remove clutter by removing axes
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # remove length ticks as they are irrelevant in this vis
        ax.get_xaxis().set_ticks([])
        hierarchy.set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k', 'b'])
        dend = dendrogram(linkage_matrix, labels=leaf_names, show_leaf_counts=False,
                          orientation='left',
                          no_labels=False,
                          above_threshold_color='grey', leaf_font_size=6)

        node_color = self.get_text_colors(dend['color_list'])
        if len(node_color) == 0:
            node_color.append('grey')
        # get coordinates of the top level nodes representing the topics
        coordinates = zip(dend['icoord'], dend['dcoord'])
        max_x, text_y_coords = self.get_top_level_nodes(coordinates)
        for i in range(0, len(topics)):
            if len(topics[i]) > 6:
                x_difference = 1.125
            else:
                x_difference = 1.095
            ax.text(max_x * x_difference, text_y_coords[i] * 0.99, topics[i],
                    color=node_color[i], fontsize=8)
            plt.plot(max_x, text_y_coords[i], marker="o", color=node_color[i])

        return fig

    @staticmethod
    def get_text_colors(color_list):
        text_color = []
        for color in color_list:
            if color not in text_color and color is not 'grey':
                text_color.append(color)
        return text_color

    @staticmethod
    def display_dendrogram():
        plt.show()

    @staticmethod
    def save_dendrogram(fig_title, fig, output_folder_path, file_name):
        fig.suptitle(fig_title, fontsize=12)
        fig.savefig(output_folder_path + "\\" + file_name[:-4] + ".png")
