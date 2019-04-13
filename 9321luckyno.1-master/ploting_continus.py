import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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


def statis(df):
    # df = df[['gender',
    # resting blood pressure 'trestbps', 90-110, 110-130, 130-150, 150-170, 170-190, 190-210 : 6
    # chol serum cholestoral in mg/dl 'chol', 100-150, 150-250, ... 550-600: 10
    # thalach maximum heart rate achieved  'thalach', 70-90, 90-110, 110-130, 130-150, 150-170, 170-190, 190-210, :7
    # oldpeak = ST depression induced by exercise relative to rest 'oldpeak', 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7: 7
    # 'ageGroup']]
    print(df['trestbps'].max())
    print(df['trestbps'].mean())
    print(df['trestbps'].min())


def plot_oldpeak(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())

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
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.5, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(8, 10))

    tmp_male = [0 for i in range(size)]
    tmp_female = [0 for i in range(size)]

    male_lg = []
    male_lg_lb = []
    female_lg = []
    female_lg_lb = []

    for cont in contents.keys():
        kind_ = contents[cont]

        male = list(kind_[1].values())
        female = list(kind_[0].values())

        m_p = plt.bar(x - width / 2, male, width=width, label=legend_[cont], bottom=tmp_male, alpha=0.6,
                      tick_label=[genders_label[1] for _ in range(size)], )
        male_lg.append(m_p)
        male_lg_lb.append(legend_[cont])
        f_p = plt.bar(x + width / 2, female, width=width, label=legend_[cont], bottom=tmp_female, alpha=0.6,
                      tick_label=[genders_label[0] for _ in range(size)], )
        female_lg.append(f_p)
        female_lg_lb.append(legend_[cont])

        for i in range(size):
            tmp_male[i] += male[i]
            tmp_female[i] += female[i]

    plt.xlabel(u'Age section', fontsize=13, labelpad=10)
    plt.ylabel(u'Amount', fontsize=13, labelpad=10)
    x_ticks = [genders_label[0] + " " + genders_label[1] + '\n' + item for item in age_section.values()]
    plt.xticks(x, x_ticks, fontsize=10, )

    # plt.legend(ncol=size)
    l1 = plt.legend(male_lg, male_lg_lb, loc='upper left', ncol=2)
    l2 = plt.legend(female_lg, female_lg_lb, loc='upper right', ncol=2)
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.title(title, size=18)
    # plt.show()
    plt.savefig(continual_fold + title)


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
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.5, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(8, 8))

    tmp_male = [0 for i in range(size)]
    tmp_female = [0 for i in range(size)]

    male_lg = []
    male_lg_lb = []
    female_lg = []
    female_lg_lb = []

    for cont in contents.keys():
        kind_ = contents[cont]

        male = list(kind_[1].values())
        female = list(kind_[0].values())

        m_p = plt.bar(x - width / 2, male, width=width, label=legend_[cont], bottom=tmp_male, alpha=0.6,
                      tick_label=[genders_label[1] for _ in range(size)], )
        male_lg.append(m_p)
        male_lg_lb.append(legend_[cont])
        f_p = plt.bar(x + width / 2, female, width=width, label=legend_[cont], bottom=tmp_female, alpha=0.6,
                      tick_label=[genders_label[0] for _ in range(size)], )
        female_lg.append(f_p)
        female_lg_lb.append(legend_[cont])

        for i in range(size):
            tmp_male[i] += male[i]
            tmp_female[i] += female[i]

    plt.xlabel(u'Age section', fontsize=13, labelpad=10)
    plt.ylabel(u'Amount', fontsize=13, labelpad=10)
    x_ticks = [genders_label[0] + " " + genders_label[1] + '\n' + item for item in age_section.values()]
    plt.xticks(x, x_ticks, fontsize=10, )

    # plt.legend(ncol=size)
    l1 = plt.legend(male_lg, male_lg_lb, loc='upper left', ncol=2)
    l2 = plt.legend(female_lg, female_lg_lb, loc='upper right', ncol=2)
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.title(title, size=18)
    # plt.show()
    plt.savefig(continual_fold + title)


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
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.5, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(8, 10))

    tmp_male = [0 for i in range(size)]
    tmp_female = [0 for i in range(size)]

    male_lg = []
    male_lg_lb = []
    female_lg = []
    female_lg_lb = []

    for cont in contents.keys():
        kind_ = contents[cont]

        male = list(kind_[1].values())
        female = list(kind_[0].values())

        m_p = plt.bar(x - width / 2, male, width=width, label=legend_[cont], bottom=tmp_male, alpha=0.6,
                      tick_label=[genders_label[1] for _ in range(size)], )
        male_lg.append(m_p)
        male_lg_lb.append(legend_[cont])
        f_p = plt.bar(x + width / 2, female, width=width, label=legend_[cont], bottom=tmp_female, alpha=0.6,
                      tick_label=[genders_label[0] for _ in range(size)], )
        female_lg.append(f_p)
        female_lg_lb.append(legend_[cont])

        for i in range(size):
            tmp_male[i] += male[i]
            tmp_female[i] += female[i]

    plt.xlabel(u'Age section', fontsize=13, labelpad=10)
    plt.ylabel(u'Amount', fontsize=13, labelpad=10)
    x_ticks = [genders_label[0] + " " + genders_label[1] + '\n' + item for item in age_section.values()]
    plt.xticks(x, x_ticks, fontsize=10, )

    # plt.legend(ncol=size)
    l1 = plt.legend(male_lg, male_lg_lb, loc='upper left', ncol=2)
    l2 = plt.legend(female_lg, female_lg_lb, loc='upper right', ncol=2)
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.title(title, size=18)
    # plt.show()
    plt.savefig(continual_fold + title)


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
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.5, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(8, 8))

    tmp_male = [0 for i in range(size)]
    tmp_female = [0 for i in range(size)]

    male_lg = []
    male_lg_lb = []
    female_lg = []
    female_lg_lb = []

    for cont in contents.keys():
        kind_ = contents[cont]

        male = list(kind_[1].values())
        female = list(kind_[0].values())

        m_p = plt.bar(x - width / 2, male, width=width, label=legend_[cont], bottom=tmp_male, alpha=0.6,
                      tick_label=[genders_label[1] for _ in range(size)], )
        male_lg.append(m_p)
        male_lg_lb.append(legend_[cont])
        f_p = plt.bar(x + width / 2, female, width=width, label=legend_[cont], bottom=tmp_female, alpha=0.6,
                      tick_label=[genders_label[0] for _ in range(size)], )
        female_lg.append(f_p)
        female_lg_lb.append(legend_[cont])

        for i in range(size):
            tmp_male[i] += male[i]
            tmp_female[i] += female[i]

    plt.xlabel(u'Age section', fontsize=13, labelpad=10)
    plt.ylabel(u'Amount', fontsize=13, labelpad=10)
    x_ticks = [genders_label[0] + " " + genders_label[1] + '\n' + item for item in age_section.values()]
    plt.xticks(x, x_ticks, fontsize=10, )

    l1 = plt.legend(male_lg, male_lg_lb, loc='upper left', ncol=2)
    l2 = plt.legend(female_lg, female_lg_lb, loc='upper right', ncol=2)
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.title(title, size=18)
    plt.savefig(continual_fold + title)


if __name__ == '__main__':
    discrete_fold = r'discrete/'
    continual_fold = r'continual/'
    diet_path = r'cleanedProjectData.csv'
    df = load_data(diet_path)
    df = age_group(df)
    df = df[['gender', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ageGroup']]
    print(df[:10])

    statis(df)

    genders_label = {1: 'Male', 0: 'Female'}
    age_section = {0: '<=44', 1: '45-59', 2: '60-74', 3: '>=75'}

    ## start plotting
    plot_trestbps(df)
    plot_chol(df)
    plot_thalach(df)
    plot_oldpeak(df)
