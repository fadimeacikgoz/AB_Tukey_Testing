# İŞ PROBLEMİ
# Bir fast food zinciri menüsüne yeni bir ürün eklemeyi planlıyor
# Bu yeni ürün tanıtımı için üç olası pazarlama kampanyası konusunda ise kararsızlar
# Hangi promosyonun satışlar üzerinde en büyük etkiye sahip oldugunu belirlemek
# için yeni çıkarılan ürün rasgele secilmiş bir kaç markette tanıtılıyor
# Farklı lokasyonlardaki tanıtılan bu promosyonlar ile yeni ürünün haftalık satışları ilk dört hafta boyunca kaydedilmiştir



#GÖREV
# A/B testi sonuclarını degerlendiriniz ve hangi pazarlama stratejisini en iyi sonucu verdiğine karar veriniz

# Değişkenler
# MarketID: unique identifier for market
# MarketSize: size of market area by sales
# LocationID: unique identifier for store location
# AgeOfStore: age of store in years
# Promotion: one of three promotions that were tested
# week: one of four weeks when the promotions were run
# SalesInThousands: sales amount for a specific LocationID, Promotion, and week


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal

plt.style.use('fivethirtyeight')
df_ = pd.read_csv("/Users/fadimeacikgoz/PycharmProjects/pythonProject/Measurement Problems/measurement_problems/datasets/WA_Marketing-Campaign.csv.xls")
df = df_.copy()

df.head()
df.describe().T



# Anova (Analysis of Variance)¶

# Hipotezler
# H0 = m1=m2=m3(gruplar arasında fark yoktur)
# H1 = ... fark vardır

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal


fig, axs = plt.subplots(1,3,figsize=(15,5))

qqplot(np.array(df.loc[(df["Promotion"] == 1), "SalesInThousands"]), line="s", ax=axs[0])
qqplot(np.array(df.loc[(df["Promotion"] == 2), "SalesInThousands"]), line="s", ax=axs[1])
qqplot(np.array(df.loc[(df["Promotion"] == 3), "SalesInThousands"]), line="s", ax=axs[2])

axs[0].set_title("Promotion 1")
axs[1].set_title("Promotion 2")
axs[2].set_title("Promotion 3")

plt.show()

market_size = df["MarketSize"].unique()

for market_size in market_size:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    qqplot(np.array(df.loc[(df["Promotion"] == 1) & (df["MarketSize"] == market_size), "SalesInThousands"]), line="s",
           ax=axs[0])
    qqplot(np.array(df.loc[(df["Promotion"] == 2) & (df["MarketSize"] == market_size), "SalesInThousands"]), line="s",
           ax=axs[1])
    qqplot(np.array(df.loc[(df["Promotion"] == 3) & (df["MarketSize"] == market_size), "SalesInThousands"]), line="s",
           ax=axs[2])

    axs[0].set_title("Promotion 1")
    axs[1].set_title("Promotion 2")
    axs[2].set_title("Promotion 3")

    fig.suptitle(f"QQ-Plot by Sales & Promotion Types - {market_size} Market Size")
    plt.show()

# 1-) Normality assumption - Shapiro Wilk Test
for promotion in list (df["Promotion"].unique()):
    pvalue = shapiro(df.loc[df["Promotion"] == promotion, "SalesInThousands"])[1]
    print("Promotion:", promotion, "p-value: %.4f" % (pvalue))


# 2-) Variance Homogeneity - Levene Test
test_stat, pvalue = levene(df.loc[df["Promotion"] == 1, "SalesInThousands"],
                           df.loc[df["Promotion"] == 2, "SalesInThousands"],
                           df.loc[df["Promotion"] == 3, "SalesInThousands"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


df.groupby("Promotion").agg({"SalesInThousands":["count","mean","median"]})

df.groupby(["MarketSize","Promotion"]).agg({"SalesInThousands":["count","mean","median"]})


#3-)  ANOVA Test - Kruskal Wallis

kruskal(df.loc[df["Promotion"] == 1, "SalesInThousands"],
        df.loc[df["Promotion"] == 2, "SalesInThousands"],
        df.loc[df["Promotion"] == 3, "SalesInThousands"])


# 4-) Tukey Test (Fark hangi gruptan kaynaklanıyor )

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["SalesInThousands"], df["Promotion"])
tukey = comparison.tukeyhsd(0.05)
print(comparison.tukeyhsd(0.05))

#  Multiple Comparison of Means - Tukey HSD, FWER=0.05
# =====================================================
# group1 group2 meandiff p-adj   lower    upper  reject
# -----------------------------------------------------
#      1      2 -10.7696    0.0 -14.7738 -6.7654   True
#      1      3  -2.7345 0.2444  -6.7388  1.2697  False
#      2      3   8.0351    0.0   4.1208 11.9493   True
# -----------------------------------------------------

# Baktıgımızda 1 - 2 arasında anlamlı bir farklılık var
# 2 - 3 arasındada anlamlı bir farklılık var
# 1 - 3 arasında anlamlı bir farklılık yok (ortlamalar birbirinden farklı degil)
# 2 ' nin ayrıştıgını görüyoruz ortalamalara baktıgımızda daha düşük kalıyor 2 yi eleyebiliriz. 1-3 baktıgımızda ortalamlar
# birbirine yakın 1-3 arasında örneklendirmeyi artırarak  yine test edebiliriz .Veri attırılamıyorsa ortalamları yüksek
# olan 1 'i seceriz.


df.groupby("Promotion").agg({"SalesInThousands":["count","mean","median"]})
#           SalesInThousands
#                      count     mean   median
# Promotion
# 1                      172 58.09901 55.38500
# 2                      188 47.32941 45.38500
# 3                      188 55.36447 51.16500