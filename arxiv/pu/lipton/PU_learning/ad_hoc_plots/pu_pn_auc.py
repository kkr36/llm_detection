from matplotlib import pyplot as plt

if __name__ == "__main__":
    # pu_aucs = [.9404076949231625, .9378174341304066, .9342347000134688, .9286913073583898, .9229459117445681, .9195056557181616, .9108464743082564]
    # pn_aucs = [.9413764526287522, .9284847305786137, .9219047532024235, .9086150150048834, .8925361159197351, .900539688471243, .8688789169403675]
    # pcts = [0, .1, .2, .3, .4, .5, .6]
    pu_aucs = [.946, .932, .898]
    pn_aucs = [.948, .923, .874]
    pcts = [2012, 2018, 2020]

    # purem_aucs = [.9404076949231625, .9396746275396957, .9390406725150863, .9367739071842749, .9344651156707116, .9319159021203929, .9298438980347993]
    # pnrem_aucs = [.9413764526287522, .9387650463177308, .9400209503414038, .9388982035925705, .9352448767754549, .9362563217795763, .9313200004718473]

    plt.plot(pcts, pn_aucs, color="blue", label="PN")
    plt.plot(pcts, pu_aucs, color="red", label="PU")
    # plt.plot(pcts, pnrem_aucs, color="blue", linestyle="--", label="PN, perfect P/N split")
    # plt.plot(pcts, purem_aucs, color="red", linestyle="--", label="PU, perfect P/N split")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pu_pn_auc_time.pdf", format='pdf', bbox_inches="tight")