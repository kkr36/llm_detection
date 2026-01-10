from matplotlib import pyplot as plt

aucs = [.9313296457524339, .92195892383662, .9156110716744541, .9043238718236616, .9026938283680004, .8926155975202108]
year = [2010, 2012, 2014, 2016, 2018, 2020]

plt.plot(year, aucs)
plt.ylabel("AUC")
plt.xlabel("year")
plt.savefig("pu_over_time.pdf", format="pdf")