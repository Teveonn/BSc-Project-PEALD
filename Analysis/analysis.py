import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import os 
import csv

make_plots = True
show_plots = False
save_plots = True
legend_position = 'upper left'


# define reations to search for 
# add +(wavelength) for the reactions the produce multiple wavelengths to specify
desired_reations = [

    "r550: O_5s -> O+(135.6_nm)",
    "r551: O_5s -> O+(135.9_nm)",
    "r552: O_3s -> O+(130.2_nm)",
    "r553: O_3s -> O+(130.5_nm)",
    "r554: O_3s -> O+(130.6_nm)",
    "r549: O_1s -> O_1d+(557.7_nm)",
    "r555: O_5p -> O_5s+(777.2_nm)",
    "r556: O_5p -> O_5s+(777.4_nm)",
    "r557: O_5p -> O_5s+(777.5_nm)",
    "r558: O_3p -> O_3s+(844.6_nm)",
    "r559: O_3p -> O_3s+(844.6_nm)",
    "r560: O_3p -> O_3s+(844.7_nm)"
]

independent_var_name = "Power (W)"
independent_var_values= [100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]


path = os.path.dirname(os.path.abspath(__file__))

os.makedirs("initial_plots", exist_ok=True)


data_files_names = (os.listdir(path + "/data_to_analyse"))

temp_filename_breakpoint = independent_var_name.split("(")[1][0].lower()

print(temp_filename_breakpoint)

sorted_files = sorted(data_files_names, key=lambda x: float(x.split(temp_filename_breakpoint)[0][:]))
print("hi")


data_files = []

for data_file_name in sorted_files:
    data_files.append(pd.read_csv(path + "/data_to_analyse/" + data_file_name +"/K.csv"))





extracted_data = []


for df in data_files:


    desired_reactions_data = []
    for reaction in desired_reations:
        for header in df:
            reaction = reaction.split("+")[0]
            if header == reaction:
                desired_reactions_data.append(df[reaction])

    extracted_data.append(desired_reactions_data)

# index [] refers to each run
# index [][] refers to each reaction in each run


n_reactions_to_graph = (len(extracted_data[0]))
n_datapoints_to_graph = (len(extracted_data))


# find stabilised values to plot 
# index [] refers to each reaction
# index [][] refers to each datapoint
all_stabilised_values = []
for i in range(n_reactions_to_graph):
    stabilised_values_per_reaction = []
    for j in range(n_datapoints_to_graph):
        stabilised_values_per_reaction.append(extracted_data[j][i].iloc[-1])

    all_stabilised_values.append(stabilised_values_per_reaction)

print(extracted_data)


# make approprate graph titles
graph_titles = []
for i in desired_reations:
    temp_graph_title = i.split(":")[1]
    temp_graph_title = temp_graph_title.replace("+"," ")
    graph_titles.append(temp_graph_title)


if make_plots:

    """
    # Plot individually
    for reaction_data in all_stabilised_values:
        plt.plot(independent_var_values, reaction_data,"o", color = "red")
    #   plt.plot(independent_var_values, reaction_data,"-" , color = "blue")
        plt.xlabel(independent_var_name)
        plt.ylabel(graph_titles[all_stabilised_values.index(reaction_data)] + " flux")
        plt.title(graph_titles[all_stabilised_values.index(reaction_data)] + " with changing " + independent_var_name)


        if save_plots:
            plt.savefig(graph_titles[all_stabilised_values.index(reaction_data)] + ".png")

        plt.show()
     """


    # plot all of the reactions on the same graph, can be both with and without a line on the graph
    def plot_data(index_range,all_stabilised_values, independent_var_values, independent_var_name, graph_titles, add_lines,Title):

        if index_range[1] - index_range[0] != 1:
            multiple_plots = True
        else:
            multiple_plots = False

        # create a colormap
        cmap = plt.get_cmap('plasma')

        # create a scalar mappable to map the colormap to the data points
        norm = colors.Normalize(vmin=0, vmax=len(all_stabilised_values[index_range[0]:index_range[1]]))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # create the plot
        fig, ax = plt.subplots()
        for i, reaction_data in enumerate(all_stabilised_values[index_range[0]:index_range[1]]):
            color = sm.to_rgba(i)
            ax.scatter(independent_var_values, reaction_data, color=color, label=graph_titles[index_range[0]+i])
            if add_lines == True:
                ax.plot(independent_var_values, reaction_data, color=color, linestyle='--')

        # add a legend
        if multiple_plots == True:
            ax.legend(loc=legend_position, fontsize = 8)
            if Title == "none":
                ax.set_title("Flux of different reactions with changing " + independent_var_name)
            else:
                ax.set_title(Title)
        else:
            ax.set_title("Flux of " + graph_titles[index_range[0]] + " with changing " + independent_var_name)

        # set the labels and title
        ax.set_xlabel(independent_var_name)
        ax.set_ylabel("Flux")


        if save_plots:
           if multiple_plots == True:
                if add_lines:
                    plt.savefig(f"initial_plots/Flux of reaction_{str(index_range[0])}_to_{str(index_range[1]-1)}_lines_{independent_var_name}.png", bbox_inches="tight", dpi=400)
                else:
                    plt.savefig(f"initial_plots/Flux of reaction_{str(index_range[0])}_to_{str(index_range[1]-1)}_{independent_var_name}.png", bbox_inches="tight", dpi=400)
           else:
                if add_lines:
                    plt.savefig(f"initial_plots/Flux of reaction_{str(index_range[0])}_lines_{independent_var_name}.png", bbox_inches="tight", dpi=400)
                else:
                    plt.savefig(f"initial_plots/Flux of reaction_{str(index_range[0])}_{independent_var_name}.png", bbox_inches="tight", dpi=400)
        # show the plot
        if show_plots == True:
            plt.show()
        plt.close()

    # index range for reaction groups


    O_5s = [0,2]
    O_3s = [2,5]
    O_1s = [5,6]
    O_5p = [6,9]
    O_3p = [9,12]

    VUVphotons = [0,5]
    nonVUVphotons = [5,12]

    all_reactions = [0,12]

    for i in range(len(desired_reations)):
        plot_data([i,i+1], all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")

    plot_data(O_1s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(O_1s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")

    plot_data(O_5s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(O_5s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")

    plot_data(O_3s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(O_3s, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")

    plot_data(O_5p, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(O_5p, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")

    plot_data(O_3p, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(O_3p, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")
    
    plot_data(VUVphotons, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "Flux of VUV photons with changing " + independent_var_name)
    plot_data(VUVphotons, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "Flux of VUV photons with changing " + independent_var_name)

    plot_data(nonVUVphotons, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "Flux of non VUV photons with changing " + independent_var_name)
    plot_data(nonVUVphotons, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "Flux of non VUV photons with changing " + independent_var_name)

    plot_data(all_reactions, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, False, "none")
    plot_data(all_reactions, all_stabilised_values, independent_var_values, independent_var_name, graph_titles, True, "none")



all_stabilised_values_transposed = map(list, zip(*all_stabilised_values))


with open('analysed_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write the headers
    csvfile_headers = [independent_var_name] + graph_titles
    writer.writerow(csvfile_headers)

    for i, row in enumerate(all_stabilised_values_transposed):
        writer.writerow([independent_var_values[i], *row])


    # write the data
"""     for i, reaction_title in enumerate(reaction_titles):
        writer.writerow([reaction_title, stabilized_values[i], independent_var_values[i]]) """

