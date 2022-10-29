data perf_ran;
infile "/home/n3401172280/summer2020/ML_project/perf_random_internal.csv" firstobs=2 dsd;
input algorithm $ datasize auc rnum;
if datasize <=5 then stable = 1;
else stable = 2;
run;
proc print data=perf_ran(obs=10);run;


proc glm data=perf_ran;
class algorithm datasize rnum;
model auc = algorithm datasize rnum;
means datasize/bon;
run;

/* One-way ANOVA test (terms including 'algorithm', 'datasize', 'random_number') the difference 
among the each training datasets, after bonferroni correction, we know that there is no difference 
from 1 to 5, and 5 to 10. Then we separated them as two phase: Unstable and Stable.  */

proc glm data=perf_ran;
where stable = 1;
class algorithm;
model auc = algorithm;
means algorithm / bon clm cldiff;
run;

proc glm data=perf_ran;
where stable = 2;
class algorithm;
model auc = algorithm;
means algorithm / bon clm cldiff;
run;


*external validation comparison part;
data perf_ran;
infile "/home/n3401172280/summer2020/ML_project/perf_random_external.csv" firstobs=2 dsd;
input algorithm $ datasize auc rnum dataset $;
run;

proc glm data=perf_ran;
where dataset = "liz";
class algorithm;
model auc = algorithm;
means algorithm / bon clm cldiff;
run; *3star rf - gb, lr - rf, ns gb-lr;

proc glm data=perf_ran;
where dataset = "cstb";
class algorithm;
model auc = algorithm;
means algorithm / bon clm cldiff;
run; *3star rf - gb, lr - rf, ns gb-lr;

proc glm data=perf_ran;
where dataset = "dr";
class algorithm;
model auc = algorithm;
means algorithm / bon clm cldiff;
run; * 3star rf - gb, lr - gb, ns lr - rf;



