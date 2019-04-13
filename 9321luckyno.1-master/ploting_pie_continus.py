import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager as fm

pd.set_option('display.max_columns', None)

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')


def load_data(diet_path):
    return pd.read_csv(diet_path, )


def age_group(df):
    def assign_(row):
        age = float(row['age'])
        if age <= 44.:
            a_g = 0
        elif age > 44. and age <= 59.:
            a_g = 1
        elif age > 59. and age <= 74.:
            a_g = 2
        else:
            a_g = 3

        return a_g

    df['ageGroup'] = df.apply(assign_, axis=1)
    df.drop('age', axis=1, inplace=True)
    return df


def plot_oldpeak(df):
    age_sec_id = list(age_section.keys())
    age_sec_names = list(age_section.values())

    title = 'oldpeak = ST depression induced by exercise relative to rest'
    legend_ = {
        0.0: "0-1", 1.0: "1-2", 2.0: "2-3", 3.0: "3-4", 4.0: "4-5", 5.0: "5-6", 6.0: "6-7",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'oldpeak'

        def assign_(row):
            t_r = float(row[target_row])
            if t_r <= 1.:
                a_g = 0
            elif t_r > 1 and t_r <= 2:
                a_g = 1
            elif t_r > 2 and t_r <= 3:
                a_g = 2
            elif t_r > 3. and t_r <= 4.:
                a_g = 3
            elif t_r > 4. and t_r <= 5.:
                a_g = 4
            elif t_r > 5. and t_r <= 6.:
                a_g = 5
            elif t_r > 6. and t_r <= 7.:
                a_g = 6
            else:
                a_g = 7

            return a_g

        df[target_row] = df.apply(assign_, axis=1)

        kinds = list(set(df[target_row].tolist()))
        print(kinds)

        contents = dict()
        for kind in kinds:
            age_sec_id_content = dict({1: dict({id: 0 for id in age_sec_id}), 0: dict({id: 0 for id in age_sec_id})})
            contents[kind] = age_sec_id_content

        df_ = df[['ageGroup', 'gender', target_row]]

        for kind in kinds:
            ag_dfs = df_.groupby(['ageGroup'])
            for name_, sub_df in ag_dfs:
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)

    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(133)
    ax3 = plt.subplot(132)
    male_sizes = []
    female_sizes = []
    labels = list(legend_.values())
    for cont in contents.keys():
        kind_ = contents[cont]
        male = sum(list(kind_[1].values()))
        male_sizes.append(male)
        female = sum(list(kind_[0].values()))
        female_sizes.append(female)

    a = np.random.rand(2, 100)
    color_vals = list(a[0])
    my_norm = mpl.colors.Normalize(0, 1)
    my_cmap = mpl.cm.get_cmap('rainbow', len(color_vals))
    colors = my_cmap(my_norm(list(a[0])))

    explode = [0.01 for i in range(0, len(contents))]
    patches1, l_text1, p_text1 = ax1.pie(male_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax1.set_title(genders_label[1])
    patches2, l_text2, p_text2 = ax2.pie(female_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax2.set_title(genders_label[0])

    proptease = fm.FontProperties()
    proptease.set_size('small')
    plt.setp(p_text1, fontproperties=proptease)
    plt.setp(l_text1, fontproperties=proptease)
    plt.setp(p_text2, fontproperties=proptease)
    plt.setp(l_text2, fontproperties=proptease)

    ax1.axis('equal')
    ax2.axis('equal')

    ax3.axis('off')
    ax3.legend(patches1, labels, loc='center')
    ax3.legend(patches2, labels, loc='center')

    plt.tight_layout()
    plt.title(title, size=20)
    plt.savefig(continual_fold + 'pie chart - ' + title, dpi=200, bbox_inches='tight')


def plot_thalach(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())

    title = 'thalach maximum heart rate achieved'
    legend_ = {
        0.0: "70-90", 1.0: "90-110", 2.0: "110-130", 3.0: "130-150", 4.0: "150-170", 5.0: "170-190", 6.0: "190-210",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'thalach'

        def assign_(row):
            t_r = float(row[target_row])
            if t_r <= 90.:
                a_g = 0
            elif t_r > 90 and t_r <= 110:
                a_g = 1
            elif t_r > 110 and t_r <= 130:
                a_g = 2
            elif t_r > 130. and t_r <= 150.:
                a_g = 3
            elif t_r > 150. and t_r <= 170.:
                a_g = 4
            elif t_r > 170. and t_r <= 190.:
                a_g = 5
            elif t_r > 190. and t_r <= 210.:
                a_g = 6
            else:
                a_g = 7

            return a_g

        df[target_row] = df.apply(assign_, axis=1)

        kinds = list(set(df[target_row].tolist()))
        print(kinds)

        contents = dict()
        for kind in kinds:
            age_sec_id_content = dict({1: dict({id: 0 for id in age_sec_id}), 0: dict({id: 0 for id in age_sec_id})})
            contents[kind] = age_sec_id_content

        df_ = df[['ageGroup', 'gender', target_row]]

        for kind in kinds:
            ag_dfs = df_.groupby(['ageGroup'])
            for name_, sub_df in ag_dfs:
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)
    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(133)
    ax3 = plt.subplot(132)
    male_sizes = []
    female_sizes = []
    labels = list(legend_.values())
    for cont in contents.keys():
        kind_ = contents[cont]
        male = sum(list(kind_[1].values()))
        male_sizes.append(male)
        female = sum(list(kind_[0].values()))
        female_sizes.append(female)

    a = np.random.rand(2, 100)
    color_vals = list(a[0])
    my_norm = mpl.colors.Normalize(0, 1)
    my_cmap = mpl.cm.get_cmap('rainbow', len(color_vals))
    colors = my_cmap(my_norm(list(a[0])))

    explode = [0.01 for i in range(0, len(contents))]
    patches1, l_text1, p_text1 = ax1.pie(male_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax1.set_title(genders_label[1])
    patches2, l_text2, p_text2 = ax2.pie(female_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax2.set_title(genders_label[0])

    proptease = fm.FontProperties()
    proptease.set_size('small')
    plt.setp(p_text1, fontproperties=proptease)
    plt.setp(l_text1, fontproperties=proptease)
    plt.setp(p_text2, fontproperties=proptease)
    plt.setp(l_text2, fontproperties=proptease)

    ax1.axis('equal')
    ax2.axis('equal')

    ax3.axis('off')
    ax3.legend(patches1, labels, loc='center')
    ax3.legend(patches2, labels, loc='center')

    plt.tight_layout()
    plt.title(title, size=20)
    plt.savefig(continual_fold + 'pie chart - ' + title, dpi=200, bbox_inches='tight')


def plot_chol(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())

    title = 'chol serum cholestoral in mg-dl'
    legend_ = {
        1.0: "100-150", 2.0: "150-200", 3.0: "200-250", 4.0: "250-300", 5.0: "300-350", 6.0: "350-400",
        7.0: "400-450", 8.0: "450-500", 9.0: "500-550", 10.0: "550-600",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'chol'

        def assign_(row):
            t_r = float(row[target_row])
            if t_r <= 150.:
                a_g = 1
            elif t_r > 150 and t_r <= 200:
                a_g = 2
            elif t_r > 200. and t_r <= 250.:
                a_g = 3
            elif t_r > 250. and t_r <= 300.:
                a_g = 4
            elif t_r > 300. and t_r <= 350.:
                a_g = 5
            elif t_r > 350. and t_r <= 400.:
                a_g = 6
            elif t_r > 400. and t_r <= 450.:
                a_g = 7
            elif t_r > 450. and t_r <= 500.:
                a_g = 8
            elif t_r > 500. and t_r <= 550.:
                a_g = 9
            elif t_r > 550. and t_r <= 600.:
                a_g = 10
            else:
                a_g = 11

            return a_g

        df[target_row] = df.apply(assign_, axis=1)

        kinds = list(set(df[target_row].tolist()))
        print(kinds)

        contents = dict()
        for kind in kinds:
            age_sec_id_content = dict({1: dict({id: 0 for id in age_sec_id}), 0: dict({id: 0 for id in age_sec_id})})
            contents[kind] = age_sec_id_content

        df_ = df[['ageGroup', 'gender', target_row]]

        for kind in kinds:
            ag_dfs = df_.groupby(['ageGroup'])
            for name_, sub_df in ag_dfs:
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)

    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(133)
    ax3 = plt.subplot(132)
    male_sizes = []
    female_sizes = []
    labels = list(legend_.values())
    for cont in contents.keys():
        kind_ = contents[cont]
        male = sum(list(kind_[1].values()))
        male_sizes.append(male)
        female = sum(list(kind_[0].values()))
        female_sizes.append(female)

    a = np.random.rand(2, 100)
    color_vals = list(a[0])
    my_norm = mpl.colors.Normalize(0, 1)
    my_cmap = mpl.cm.get_cmap('rainbow', len(color_vals))
    colors = my_cmap(my_norm(list(a[0])))

    explode = [0.01 for i in range(0, len(contents))]
    patches1, l_text1, p_text1 = ax1.pie(male_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax1.set_title(genders_label[1])
    patches2, l_text2, p_text2 = ax2.pie(female_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax2.set_title(genders_label[0])

    proptease = fm.FontProperties()
    proptease.set_size('small')
    plt.setp(p_text1, fontproperties=proptease)
    plt.setp(l_text1, fontproperties=proptease)
    plt.setp(p_text2, fontproperties=proptease)
    plt.setp(l_text2, fontproperties=proptease)

    ax1.axis('equal')
    ax2.axis('equal')

    ax3.axis('off')
    ax3.legend(patches1, labels, loc='center')
    ax3.legend(patches2, labels, loc='center')

    plt.tight_layout()
    plt.title(title, size=20)
    plt.savefig(continual_fold + 'pie chart - ' + title, dpi=200, bbox_inches='tight')


def plot_trestbps(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())

    title = 'resting blood pressure'
    legend_ = {
        1.0: "90-110", 2.0: "110-130", 3.0: "130-150", 4.0: "150-170", 5.0: "170-190", 6.0: "190-210",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'trestbps'

        def assign_(row):
            t_r = float(row[target_row])
            if t_r <= 110.:
                a_g = 1
            elif t_r > 110 and t_r <= 130:
                a_g = 2
            elif t_r > 130. and t_r <= 150.:
                a_g = 3
            elif t_r > 150. and t_r <= 170.:
                a_g = 4
            elif t_r > 170. and t_r <= 190.:
                a_g = 5
            elif t_r > 190. and t_r <= 210.:
                a_g = 6
            else:
                a_g = 7

            return a_g

        df[target_row] = df.apply(assign_, axis=1)

        kinds = list(set(df[target_row].tolist()))
        print(kinds)

        contents = dict()
        for kind in kinds:
            age_sec_id_content = dict({1: dict({id: 0 for id in age_sec_id}), 0: dict({id: 0 for id in age_sec_id})})
            contents[kind] = age_sec_id_content

        df_ = df[['ageGroup', 'gender', target_row]]

        for kind in kinds:
            ag_dfs = df_.groupby(['ageGroup'])
            for name_, sub_df in ag_dfs:
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)

    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(133)
    ax3 = plt.subplot(132)
    male_sizes = []
    female_sizes = []
    labels = list(legend_.values())
    for cont in contents.keys():
        kind_ = contents[cont]
        male = sum(list(kind_[1].values()))
        male_sizes.append(male)
        female = sum(list(kind_[0].values()))
        female_sizes.append(female)

    a = np.random.rand(2, 100)
    color_vals = list(a[0])
    my_norm = mpl.colors.Normalize(0, 1)
    my_cmap = mpl.cm.get_cmap('rainbow', len(color_vals))
    colors = my_cmap(my_norm(list(a[0])))

    explode = [0.01 for i in range(0, len(contents))]
    patches1, l_text1, p_text1 = ax1.pie(male_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax1.set_title(genders_label[1])
    patches2, l_text2, p_text2 = ax2.pie(female_sizes, explode=explode, labeldistance=10.1, autopct='%3.1f%%',
                                         shadow=False, startangle=90, pctdistance=0.6, colors=colors)
    ax2.set_title(genders_label[0])

    proptease = fm.FontProperties()
    proptease.set_size('small')
    plt.setp(p_text1, fontproperties=proptease)
    plt.setp(l_text1, fontproperties=proptease)
    plt.setp(p_text2, fontproperties=proptease)
    plt.setp(l_text2, fontproperties=proptease)

    ax1.axis('equal')
    ax2.axis('equal')

    ax3.axis('off')
    ax3.legend(patches1, labels, loc='center')
    ax3.legend(patches2, labels, loc='center')

    plt.tight_layout()
    plt.title(title, size=20)
    plt.savefig(continual_fold + 'pie chart - ' + title, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    discrete_fold = r'discrete/'
    continual_fold = r'continual/'
    diet_path = r'cleanedProjectData.csv'
    df = load_data(diet_path)
    df = age_group(df)
    df = df[['gender', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ageGroup']]

    genders_label = {1: 'Male', 0: 'Female'}
    age_section = {0: '<=44', 1: '45-59', 2: '60-74', 3: '>=75'}

    ## start plotting
    plot_trestbps(df)
    plot_chol(df)
    plot_thalach(df)
    plot_oldpeak(df)
