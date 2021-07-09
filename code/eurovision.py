from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn import preprocessing, decomposition, cluster, manifold
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_rows', None) # increase number of rows displayed
pd.set_option('display.max_columns', None)

def read_file(file_name):
    """Reads csv file and returns a pandas dataframe"""
    df = pd.read_csv(file_name)
    df_clean = df.drop('Unnamed: 0', axis=1) # remove redundant column
    return df_clean

def inspect_data(df):
    """Check for missing values, data ranges, data types and unique values
    for countries
    """
    print("Missing Values: \n", df.isnull().sum(), "\n")
    print("Summary Statistics: \n", df.describe(), "\n")
    print("Data Types: \n", df.dtypes)
    print("Unique Values: 'From' \n", df['From'].unique(), "\n")
    print("Unique Values: 'To' \n", df['To'].unique(), "\n")

def clean_names(df):
    """Remove hidden blank spaces and dashes from names in dataframe"""
    df['From'] = df['From'].apply(lambda row: re.sub('-', ' ', row.strip()))
    df['To'] = df['To'].apply(lambda row: re.sub('-', ' ', row.strip()))
    df['Points type'] = df['Points type'].apply(lambda row: row.strip())
    return df

def create_additional_columns(df):
    """Create identifier columns for each voter type and country combination
    and for voter type
    """
    df.insert(3, 'voter', df.apply(lambda row: label_voter_type(row), axis=1))
    df.insert(3, 'voter_country', df['voter'] + '_' + df['From'])
    return df

def label_voter_type(row):
    """Returns the voter type"""
    if re.search(r'televoter', row['Points type']):
        return 'televoter'
    elif re.search(r'jury', row['Points type']):
        return 'jury'

def points_diff(df, year):
    """Calculate the difference in points between televoter and jury by
    country
    """
    df_filtered_by_year = df[df.Year==year]
    df_pivot = (df_filtered_by_year.pivot(index='To',
        columns=['From','voter'], values='Points').fillna(0))
    df_diff = (df_pivot.loc[:, pd.IndexSlice[:, 'televoter']]
        - df_pivot.loc[:,pd.IndexSlice[:,'jury']].values).rename(
        columns={'televoter' : 'diff'})
    df_diff = df_diff.droplevel('voter', axis=1)
    df_diff = df_diff[df_diff.sort_index(axis=1,level=0,ascending=True).columns]
    fig, ax1 = plt.subplots(1, 1)
    sns.heatmap(df_diff
        , xticklabels=df_diff.columns
        , yticklabels=df_diff.columns
        , vmin=-12
        , vmax=12
        , cmap="vlag")
    ax1.set(xlabel="Votes From", ylabel="Votes To")
    plt.show()

def create_df_wide_by_year(df, year):
    """Adds the missing combinations of countries from/to and fills value
    with a zero"""
    df_filtered_by_year = df[df.Year==year]
    df_wide = (df_filtered_by_year
        .set_index(list(df_filtered_by_year.columns[:-1]))
        .unstack(fill_value=0)
#        .stack().reset_index() # add this to return the data to long format
        )
    # get missing list of countries with zero points
    missing_countries = list(set(df_filtered_by_year.From.unique())
        - set(df_filtered_by_year.To.unique()))
    # add missing countries to data frame
    for country in missing_countries:
        df_wide['Points',country] = 0
    # re-sort the columns in dataframe
    df_wide = df_wide[df_wide.sort_index(axis=1,
        level=[0,1],ascending=[True,True]).columns]
    np_matrix = df_wide.to_numpy()
    # get the names of the countries
    country_labels = df_wide.columns.get_level_values(1)
    # get the labels for voter_country combination
    voter_country_labels = df_wide.index.get_level_values(3)
    return df_wide, np_matrix, country_labels, voter_country_labels

def calc_cosine_similarity(np_matrix, voter_country_labels, country_labels):
    """Calculate cosine similarity matrix on voter preferences"""
    cosine_sim_matrix = cosine_similarity(np_matrix)
    cosine_sim_matrix_len = cosine_sim_matrix.shape[0]
    i1 = int(cosine_sim_matrix_len/2) # index to only select first n/2 entries
    i2 = cosine_sim_matrix_len
    # similarity scores for all voter-country combinations
    df_cosine_sim_all = pd.DataFrame(cosine_sim_matrix,
        columns=voter_country_labels,
        index=voter_country_labels)
    # similarity scores for televoters only
    df_cosine_sim_televoter_only = df_cosine_sim_all.iloc[0:i1,0:i1]
    # similarity scores for voters vs jury
    df_cosine_sim_televoter_vs_jury = df_cosine_sim_all.iloc[0:i1,i1:i2]
    # similarity scores for voters vs jury by country
    diagonals_televoter_vs_jury = ([df_cosine_sim_televoter_vs_jury.iat[n, n]
        for n in range(len(df_cosine_sim_televoter_vs_jury))])
    df_televoter_vs_jury_diagonals = pd.DataFrame(
        list(zip(list(country_labels),
        diagonals_televoter_vs_jury)), columns=['Country','Similarity'])
    return (df_cosine_sim_all, df_cosine_sim_televoter_vs_jury,
        df_cosine_sim_televoter_only, df_televoter_vs_jury_diagonals)

def plot_similarity(tlvr_matrix, tlvr_jury_matrix, tlvr_jury_tbl):
    """Plot heatmap of similarity matrix"""
    print(tlvr_jury_tbl)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(tlvr_jury_matrix
        , xticklabels=tlvr_jury_tbl.Country
        , yticklabels=tlvr_jury_tbl.Country
        , vmin=0
        , vmax=1
        , ax=ax1)
    ax1.set(xlabel="Jury Vote", ylabel="Televoter Vote")
    country_order = tlvr_jury_tbl.sort_values(by="Similarity"
        , ascending=False)['Country']
    colour_order = tlvr_jury_tbl.sort_values(by="Similarity"
        , ascending=False)['Similarity']
    sns.barplot(y="Country"
        , x="Similarity"
        , data=tlvr_jury_tbl
        , order=country_order
        , palette=mpl.cm.magma(colour_order)
        , ax=ax2)
    plt.show()

def print_top_bottom_n_similarity(df_wide, tlvr_jury_tbl, n=3):
    """Displays the voting patterns for the top/bottom n countries"""
    df_wide_reindexed = (df_wide.reset_index().drop(
        ['Edition','Year','Points type','voter','From'], axis=1))
    tbl = tlvr_jury_tbl.sort_values(by='Similarity', ascending=False)
    top_n_countries = tbl.head(n)['Country']
    bottom_n_countries = tbl.tail(n)['Country']
    print('Top {} Countries with Similar Scores'.format(n))
    print(list(top_n_countries))
    for country in top_n_countries:
        print(df_wide_reindexed[df_wide_reindexed.voter_country.str.contains(country)])
    print('Bottom {} Countries with Similar Scores'.format(n))
    print(list(bottom_n_countries))
    for country in bottom_n_countries:
        print(df_wide_reindexed[df_wide_reindexed.voter_country.str.contains(country)])

def normalize_data(np_matrix):
    """Normalize the data - center it on mean zero with unit variance"""
    scaler = preprocessing.StandardScaler()
    np_matrix_normalized = scaler.fit_transform(np_matrix)
    return np_matrix_normalized

def pca(np_matrix_normalized, np_matrix_col_names, evr=0.4):
    """Compute the principal components for plotting. Specify the required
    explained variance ratio (between 0 and 1) to compute the number of
    principal components.
    """
    pca = decomposition.PCA(n_components=evr)
    pca.fit(np_matrix_normalized)
    pca_transformed_values = pca.transform(np_matrix_normalized)
    pca_transformed_values_3d = pca_transformed_values[:,0:3]
    num_pc = pca.components_.T.shape[1]
    col_names = list(['PC'+str(i+1) for i in range(0,num_pc)])
    loadings = pd.DataFrame(pca.components_.T,
        columns=col_names,
        index=np_matrix_col_names)
#    print("Number of Principal Components: ", num_pc)
#    print("PCA Explained Variance: ", np.sum(pca.explained_variance_ratio_))
#    print("PCA Explained Variance Ratio: ", pca.explained_variance_ratio_)
#    print("PCA Loadings: ")
#    print(loadings)
    return pca_transformed_values, pca_transformed_values_3d

def tsne(np_matrix_normalized):
    """Compute tsne for plotting"""
    tsne_transformed_values = manifold.TSNE(n_components=3).fit_transform(np_matrix)
    print(tsne_transformed_values)
    return tsne_transformed_values

def k_means_clustering(np_matrix):
    """Compute clusters
    Note: flat elbow seems to suggest there's not much clustering activity
    """
    distortions = []
    K = range(2,26)
    # calculate kmeans and plot to determine number of cluster via elbow method
    for k in K:
        kmeanModel = cluster.KMeans(n_clusters=k)
        kmeanModel.fit(np_matrix)
        distortions.append(kmeanModel.inertia_)
    # plt.figure(figsize=(16,8))
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    kmeans = (cluster.KMeans(n_clusters=4, random_state=0)
        .fit(np_matrix))
    labels = kmeans.labels_
    print(labels)
    return labels

def merge_pca_k_means_to_df(df, pca_values, tsne_values, k_means_labels):
    df['pca1'] = pca_values[:,0]
    df['pca2'] = pca_values[:,1]
    df['pca3'] = pca_values[:,2]
    df['tsne1'] = tsne_values[:,0]
    df['tsne2'] = tsne_values[:,1]
    df['tsne3'] = tsne_values[:,2]
    df['kmeans_cluster'] = k_means_labels
    df = df.reset_index()
    df = df[['voter_country','voter','From','pca1','pca2','pca3','tsne1','tsne2','tsne3','kmeans_cluster']]
    print(df.head(200))
    print(df.columns)
    return df

def plot_chart(df):
    """Plot diagram"""
    #plot 2d chart
    fig, (ax1, ax2) = plt.subplots(1,2)
    sns.scatterplot(x="pca1", y="pca2", hue="kmeans_cluster", style="voter", palette="Set2", s=50, data=df, ax=ax1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    for line in range(0,df.shape[0]):
        ax1.text(df.pca1[line]+0, df.pca2[line], df.From[line], horizontalalignment='left', size='small', color='black')
    sns.scatterplot(x="tsne1", y="tsne2", hue="kmeans_cluster", style="voter", palette="Set2", s=50, data=df, ax=ax2)
    for line in range(0,df.shape[0]):
        ax2.text(df.tsne1[line]+0, df.tsne2[line], df.From[line], horizontalalignment='left', size='small', color='black')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    # #plot 3d chart
    # p = figure(title="PCA", x_axis_label='PCA1', y_axis_label='PCA2')
    # p.circle(df.pca1, df.pca2, size=2)
    # show(p)


data = read_file('../data/eurovision_1957-2021.csv')
data = clean_names(data)
data = create_additional_columns(data)

# years = [2016, 2017, 2018, 2019, 2021]
# df_cosinesim_by_year_list = []
#
# for year in years:
#     points_diff(data,year)
#     df_wide, np_matrix, country_labels, voter_country_labels = create_df_wide_by_year(data, year)
#     df_cosine_sim_all, df_cosine_sim_televoter_vs_jury, df_cosine_sim_televoter_only, df_televoter_vs_jury_diagonals = calc_cosine_similarity(np_matrix, voter_country_labels, country_labels)
#     print_top_bottom_n_similarity(df_wide, df_televoter_vs_jury_diagonals, n=3)
#     plot_similarity(df_cosine_sim_televoter_only, df_cosine_sim_televoter_vs_jury, df_televoter_vs_jury_diagonals)
#     # collect the similarity scores by country by year for temporal analysis
#     df_televoter_vs_jury_diagonals['Year'] = year
#     df_cosinesim_by_year_list.append(df_televoter_vs_jury_diagonals)
#
# #temporal analysis of results
# df_cosinesim_combined = pd.concat(df_cosinesim_by_year_list, ignore_index=True)
# country_counts = df_cosinesim_combined['Country'].value_counts().reset_index()
# country_counts.columns = ['Country', 'Count']
# country_counts_5 = list(country_counts[country_counts.Count == 5].Country)
# country_counts_4 = list(country_counts[country_counts.Count >= 4].Country)
# df_cosinesim_combined_subset = df_cosinesim_combined[df_cosinesim_combined.Country.isin(country_counts_4)]
# sns.lineplot(data=df_cosinesim_combined_subset, x="Year", y="Similarity", hue="Country")
# plt.show()



# plot pca and tsne
df_wide, np_matrix, country_labels, voter_country_labels = create_df_wide_by_year(data, year=2018)
np_matrix_normalized = normalize_data(np_matrix)
pca_values, pca_values_3d = pca(np_matrix_normalized, country_labels)
tsne_values = tsne(np_matrix)
k_means_labels = k_means_clustering(np_matrix)
data4 = merge_pca_k_means_to_df(df_wide, pca_values, tsne_values, k_means_labels)
print(data4)
plot_chart(data4)
