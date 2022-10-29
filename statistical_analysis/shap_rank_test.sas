data shap;
infile "/home/n3401172280/summer2020/ML_project/shap_comp.csv" firstobs=2 dsd;
input value cate $;
run;
proc print data=shap;run;
proc npar1way wilcoxon data=shap;
class cate;
var value;
run;*Kruskal-Wallis Test: P = 0.0039;