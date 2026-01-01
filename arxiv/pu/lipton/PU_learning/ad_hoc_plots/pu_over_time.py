from matplotlib import pyplot as plt

aucs = [.9393659499197897, .9324574334986052, .9274330740857908, .9173241940723194, .9167954531928744, .9087258328821142]
year = [2010, 2012, 2014, 2016, 2018, 2020]

plt.plot(year, aucs)
plt.ylabel("AUC")
plt.xlabel("year")
plt.savefig("pu_over_time.pdf", format="pdf")