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


def plot_cp3(df):
    age_sec_id = list(age_section.keys())

    title = 'chest pain type'
    legend_ = {
        1.0: "typical angin", 2.0: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'cp'
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

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_fbs6(df):
    age_sec_id = list(age_section.keys())

    title = 'fasting blood sugar'
    legend_ = {
        1.0: "> 120 mg/dl", 0.0: "<= 120 mg/dl"
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'fbs'
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

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_restecg7(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())

    title = 'resting electrocardiographic results'
    legend_ = {
        1.0: "abnormality", 0.0: "normal", 2.0: "probable"
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'restecg'
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

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_exang9(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())
    title = 'exercise induced angina'
    legend_ = {
        1.0: "positive", 0.0: "negative"
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'exang'
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

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_slope11(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())
    title = 'the slope of the peak exercise ST segment'
    legend_ = {
        1.0: "1", 0.0: "0", 3.0: "3", 2.0: "2",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'slope'
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

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_ca12(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())
    '''
    12. ca number of major vessels (0-3) colored by flourosopy： 4值数量，分层柱状图

    '''
    title = 'number of major vessels (0-3) colored by flourosopy'
    legend_ = {
        1.0: "1", 0.0: "0", 3.0: "3", 2.0: "2", 4.0: "4",
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'ca'
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
                # print(name_)
                # print(sub_df)
                # print()
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    # print(name_sub)
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


def plot_thal13(df):
    age_sec_names = list(age_section.values())
    age_sec_id = list(age_section.keys())
    '''
    13. thal (Thalassemia): 3 = normal; 6 = fixed defect; 7 = reversable defect :  6值数量，分层柱状图

    '''
    title = 'thal (Thalassemia)'
    legend_ = {
        3.0: "normal", 6.0: "fixed defect", 7.0: "reversable defect"
    }

    ## get data
    def get_relevant_data_row(df):
        target_row = 'thal'
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
                # print(name_)
                # print(sub_df)
                # print()
                sub_ag_dfs = sub_df.groupby(['gender'])
                for name_sub, sub_sub_df in sub_ag_dfs:
                    # print(name_sub)
                    contents[kind][name_sub][name_] = len(sub_sub_df[sub_sub_df[target_row] == kind])

        return contents

    contents = get_relevant_data_row(df)
    size = len(age_section)
    x = np.arange(size)

    total_width, n = 0.8, 2
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
    plt.savefig(discrete_fold + title)


if __name__ == '__main__':
    discrete_fold = r'discrete/'
    continual_fold = r'continual/'

    diet_path = r'cleanedProjectData.csv'
    df = load_data(diet_path)

    df = age_group(df)
    print(df[:20])

    genders_label = {1: 'Male', 0: 'Female'}
    age_section = {0: '<=44', 1: '45-59', 2: '60-74', 3: '>=75'}

    ## start plotting
    plot_cp3(df)
    plot_fbs6(df)
    plot_restecg7(df)
    plot_exang9(df)
    plot_slope11(df)
    plot_ca12(df)
    plot_thal13(df)
