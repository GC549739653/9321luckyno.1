import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def kde_target(var_name, df):

    # Pearson correlation coefficient between calculated variable and target value
    corr = df['label'].corr(df[var_name])

    # Calculate the median of heart disease
    has_disease = df.ix[df['label'] == 0, var_name].median()  # df.ix：索引函数，既可以通过行号索引，也可以通过行标签索引
    has_no_disease = df.ix[df['label'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # draw have or not have hd label
    sns.kdeplot(df.ix[df['label'] == 0, var_name], label='label == 0')
    sns.kdeplot(df.ix[df['label'] == 1, var_name], label='label == 1')

    # draw label
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution,the correlation between %s and the TARGET is %0.4f' % (var_name,var_name,corr))
    plt.legend()  # show

    # output pearson
    #print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    fff= 'The correlation between %s and the TARGET is %0.4f' % (var_name, corr)
    #print('Median value  that was not ill = %0.4f' % has_no_disease)
    #print('Median value  that was ill =     %0.4f' % has_disease)
    #return fff
def draw_pic():
    df = pd.read_csv(open('cleanedProjectData.csv'))
    df.rename(columns={"target":"label"},inplace=True)
    L=[]
    for e in df.columns:
            #print (e)
        kde_target(e,df)
        #L.append(ffff)
            #savepath =
            #plt.show()
        plt.savefig(r".\static\coef\%s.png" %e, dpi=520)
    #print(L)
draw_pic()