import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def kde_target(var_name, df):
    """
    绘制根据不同目标值着色的变量分布
    """
    # 计算变量与目标值之间的皮尔森相关系数
    corr = df['label'].corr(df[var_name])

    # 计算有无心脏病的中位数
    has_disease = df.ix[df['label'] == 0, var_name].median()  # df.ix：索引函数，既可以通过行号索引，也可以通过行标签索引
    has_no_disease = df.ix[df['label'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # 绘制有无心脏病的数据的分布
    sns.kdeplot(df.ix[df['label'] == 0, var_name], label='label == 0')
    sns.kdeplot(df.ix[df['label'] == 1, var_name], label='label == 1')

    # 标签绘制
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution,the correlation between %s and the TARGET is %0.4f' % (var_name,var_name,corr))
    plt.legend()  # 显示图例


    # 输出皮尔森相关系数
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # 输出均值
    print('Median value for {} that was not ill = %0.4f' % has_no_disease)
    print('Median value for {} that was ill =     %0.4f' % has_disease)


def get_corr():
    res={}
    df = pd.read_csv(open('cleanedProjectData.csv'))
    df.rename(columns={"target":"label"},inplace=True)
    for e in df.columns:
        if e!='label':
            res[e]=df['label'].corr(df[e])
            kde_target(e,df)
            plt.savefig(f"static/coef/{e}.png", dpi=520)
    return "\n".join(list(map(lambda x:f"{x[0]}:{x[1]:.4f}",list(res.items()))))

if __name__=="__main__":
    res={}
    df = pd.read_csv(open('cleanedProjectData.csv'))
    df.rename(columns={"target":"label"},inplace=True)
    for e in df.columns:
        if e!='label':
            kde_target(e,df)
            res[e]=df['label'].corr(df[e])
