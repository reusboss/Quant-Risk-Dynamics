'''
Id:          "$Id$"
Copyright:   Copyright (c) 2022 Bank of America Merrill Lynch, All Rights Reserved
Description:
Test:
'''
'''
This is a project that takes the input from the CDS curve and generate a
poisson process accordingly. Besides, creating two correlated poisson 
processes, through OOP Programming.
'''

# Import Packages
import time
import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson

plt.style.use('ggplot')


class UpperException(Exception):
    pass


# ******************************************************************************************#
'''
Start our project here :)
'''


# Create CDS Curve Class
class CDS_Curve:
    def __init__(self):
        pass

    def Generate(self, term, DF, cds_spread, recovery_rate):
        df = pd.DataFrame()
        df["Maturity"] = pd.Series(term)
        df["DF"] = pd.DataFrame(DF)
        df["Spread"] = pd.DataFrame(cds_spread)
        df["Recovery"] = recovery_rate
        return df
        # self.df["Hazard Rate"] = 0.0
        # self.df["Survival Prob"] = 1.0


class Random_Generator:
    def __init__(self, stats_type):
        if stats_type == "uni":
            self.rvs = np.random.uniform(0, 1, 50000)
            self.z = np.random.uniform(0, 1, 50000)
        elif stats_type == "expon":
            self.rvs = np.random.exponential(0.5, 50000)
            self.z = np.random.exponential(0.5, 50000)
        elif stats_type == "norm":
            self.rvs = np.random.normal(0, 1, 50000)
            self.z = np.random.normal(0, 1, 50000)

    def Cholesky_Decompose(self, rho):
        df_id = pd.Series(np.arange(50000))
        df = pd.DataFrame(index=df_id)
        df["rvs_1"] = pd.DataFrame(self.rvs)
        df["z"] = pd.DataFrame(self.z)
        df["rvs_2"] = rho * df["rvs_1"] + np.sqrt(1 - rho ** 2) * df["z"]
        print("The correlation between rvs_1 and z is: ", df["rvs_1"].corr(df["z"]))
        print("The correlation between rvs_1 and rvs_2 is: ", df["rvs_1"].corr(df["rvs_2"]))
        print("Rho is: ", rho)
        return df


# Create a class for poisson processes
class Poisson_Distribution:
    '''
    lambda is the mean of the occurrences and k is an interval that we want to
    know to test and simulate different points performances according to lambda
    **CDF = F(x); PMF = F'(x) = f(x)
    '''

    def __init__(self, lambd, k_left, k_right):
        self.lambd = lambd
        self.k = np.linspace(k_left, k_right, 100)

    def generate_pmf(self):  # Probability mass function
        pmf = poisson.pmf(self.k, mu=self.lambd)
        pmf = np.round(pmf, 7)
        return pmf

    def generate_cdf(self):  # Cumulative density function
        cdf = poisson.cdf(self.k, mu=self.lambd)
        cdf = np.round(cdf, 7)
        return cdf

    def plot(self, data):
        plt.plot(self.k, data, marker='o', label="Lambda = " + str(self.lambd))


class Poisson_Process:
    '''
    Poisson Process is simulating the homogeneous/non-homogeneous arrivals based on exponential distributions, according to the documents listed below:
    Homogeneous Poisson Process: https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasA.pdf
    Non-homogeneous Poisson Process: https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasB.pdf
    Simply, we could simulate a constant rate function that is homogeneous poisson process, or we could also simulate a time-dependent poisson process,
    of which the step function is not homogeneously upwards increasing.
    '''

    def __init__(self, cds_data):
        self.df = cds_data

    '''
    Hazard rate function is a discrete function that is not differentiable but integrable, 
    so as step function, it should return values according to the t conditions
    '''

    def lambda_t(self, t):
        if t == 0:
            return 0
        else:
            for i in range(1, len(self.df["Maturity"])):
                if t <= self.df["Maturity"][i] and t > self.df["Maturity"][i - 1]:
                    return self.df["Hazard Rate"][i]

    '''
    Generating inhomogeneous poisson process using thinning algorithm
    '''

    def inhomogeneous_simulation(self, steps):
        lambd_plus = self.df["Hazard Rate"].max()
        # Simulate homogeneous part first
        m = int(30000 * steps * lambd_plus)
        u = np.random.uniform(0, 1, m)
        t = (-1 / lambd_plus) * np.log(1 - u) / 365
        s = np.cumsum(t)
        s = np.insert(s, 0, 0)
        s = list(filter(lambda s: s < steps, s))
        s.append(steps)
        Nstar = len(s)
        # Sample using appropriate probability
        w = np.random.uniform(0, 1, Nstar)
        Ind = np.where((w <= [self.lambda_t(ds) / lambd_plus for ds in s]) == False, 0, 1)
        Nt = np.arange(0, sum(Ind))
        J = [i for i in range(len(Ind)) if Ind[i] == 1]
        s = [s[i] for i in J]
        s = np.insert(s, 0, 0)
        Nt = np.insert(Nt, 0, 0)
        return Nt, s

    def inhomogeneous_validation(self, steps):
        lambd_plus = self.df["Hazard Rate"].max()
        # Simulate homogeneous part first
        m = int(30000 * steps * lambd_plus)
        u = np.random.uniform(0, 1, m)
        t = (-1 / lambd_plus) * np.log(1 - u) / 365
        s = np.cumsum(t)
        s = np.insert(s, 0, 0)
        s = list(filter(lambda s: s < steps, s))
        s.append(steps)
        Nstar = len(s)
        # Sample using appropriate probability
        w = np.random.uniform(0, 1, Nstar)
        Ind = np.where((w <= [self.lambda_t(ds) for ds in s] / lambd_plus) == False, 0, 1)
        Nt = np.arange(0, sum(Ind))
        J = [i for i in range(len(Ind)) if Ind[i] == 1]
        s = [s[i] for i in J]
        s = list(filter(lambda s: s > 5 and s <= 10, s))
        return len(s)

    '''
    Here are two ways of simulating stepwise cumulative numbers of poisson process: Direct cumulative via Numpy.cumsum and Inverse calculation of CDF of generating rvs
    '''

    # Method 1
    def step_cumulative(self, lambd, steps, paths):
        sim_df = pd.DataFrame()
        sim_df["Steps"] = pd.Series(np.arange(steps))
        sim_df.set_index("Steps", inplace=True)
        for i in range(paths):
            y = np.cumsum(np.random.poisson(lambd, size=steps))
            sim_df["Sim_" + str(i + 1)] = pd.DataFrame(y)
            plt.plot(sim_df.index, sim_df["Sim_" + str(i + 1)], label="\u03BB = " + str(lambd))
        # plt.xlabel("Simulation Steps")
        # plt.ylabel("Number of Occurrences")
        # plt.title("Homogeneous Poisson Process (lambda = " + str(lambd) + ")")
        # plt.show()
        # print(sim_df.T)

    # Method 2
    def homogeneous_simulation(self, lambd, steps, paths):
        # np.random.seed(0)
        for i in range(paths):
            u = np.random.uniform(0, 1, 100)
            t = (-1 / lambd) * np.log(1 - u)
            s = np.cumsum(t)
            s = np.insert(s, 0, 0)
            s = list(filter(lambda s: s < steps, s))
            s.append(steps)
            Nt = np.arange(0, len(s))
            plt.step(s, Nt, label="\u03BB = " + str(lambd))

    def poisson_pdf(self, lambd, steps, paths):
        list_store = []
        for i in range(paths):
            u = np.random.uniform(0, 1, 10000)
            t = (-1 / lambd) * np.log(1 - u)
            s = np.cumsum(t)
            s = np.insert(s, 0, 0)
            s = list(filter(lambda s: s < steps, s))
            s.append(steps)
            Nt = np.arange(0, len(s))
            list_store.append(Nt[-1])
        return list_store

    def compare_interval(self, lambd, steps, paths):
        list_store1 = []
        list_store2 = []
        for i in range(paths):
            u = np.random.uniform(0, 1, 1000)
            t = (-1 / lambd) * np.log(1 - u)
            s = np.cumsum(t)
            s = np.insert(s, 0, 0)
            s1 = list(filter(lambda s: s <= 700 and s >= 0, s))
            s2 = list(filter(lambda s: s <= 1000 and s >= 300, s))
            list_store1.append(len(s1))
            list_store2.append(len(s2))
        return list_store1, list_store2


'''
Non-bootstrapping Algo just using cumulative sum and continuously compunding payment assumption:
'''


def hazard_rate(data):
    for i in range(1, len(data)):
        cum_sum = data["Hazard Rate"][:i].sum()
        data["Hazard Rate"][i] = np.round((1 - np.exp(
            -data["DF"][i] * data["Spread"][i] * data["Maturity"][i] / (1 - data["Recovery"][i]))) - cum_sum, 4)
        cum_sum_surv = data["Hazard Rate"][:i + 1].sum()
        data["Survival Prob"][i] = 1 - cum_sum_surv
    return data


'''
This is a bootstrap algo to recursively calculate the default intensity from CDS Spread (Actually Using)
'''


def bootstrap_algo(data):
    RR = data.at[0, "Recovery"]
    L = 1 - RR
    # Calculate the Survival prob
    for row, col in data.iterrows():
        if row == 0:
            data["Survival Prob"][row] = 1.0
        if row == 1:
            data["Survival Prob"][row] = L / (
                        data["Spread"][row] * (data["Maturity"][row] - data["Maturity"][row - 1]) + L)
        if row > 1:
            terms = 0
            for j in range(1, row):
                term = data["DF"][j] * (L * data["Survival Prob"][j - 1] - (
                            L + (data["Maturity"][j] - data["Maturity"][j - 1]) * data["Spread"][row]) *
                                        data["Survival Prob"][j])
                terms += term
            divider = data["DF"][row] * (L + (data["Maturity"][row] - data["Maturity"][row - 1]) * data["Spread"][row])
            term1 = terms / divider
            term2 = (data["Survival Prob"][row - 1] * L) / (
                        L + (data["Maturity"][row] - data["Maturity"][row - 1]) * data["Spread"][row])
            data["Survival Prob"][row] = term1 + term2
    # Calculate the Default Intensity
    for row in range(len(data)):
        if row == 0:
            data["Hazard Rate"][row] = 0.0
        else:
            previous = data["Cumulative Hazard"][:row].sum()
            data["Hazard Rate"][row] = np.round(
                (np.log(data["Survival Prob"][row]) + previous) / (data["Maturity"][row - 1] - data["Maturity"][row]),
                4)
        if row >= 1:
            data["Cumulative Hazard"][row] = data["Hazard Rate"][row] * (
                        data["Maturity"][row] - data["Maturity"][row - 1])
    return data[["Maturity", "DF", "Spread", "Recovery", "Survival Prob", "Hazard Rate"]]


# ***********************************************************************#
# Starting the Engine here
def main():
    '''
    The assumption is taking the LIBOR Rate = 3.5% to calculate the discount factor, and to simplify, we take spreads values directly instead of assuming
    protection or premium leg and paying frequencies. We wanted to derive Hazard Rate and its term structure from the CDS curve using Bootstrapping
    '''
    cds = pd.DataFrame()
    cds = pd.read_csv('C:\\Users\\zklhvo0\\Desktop\\CDS_Spreads.csv', header=0)
    # cds = CDS_Curve().Generate([0,1,2,3,5,7,10],[0,0.965605,0.932394,0.900325,0.839457,0.782705,0.704688],[0,110,120,140,150,160,165],0.4)
    cds["Spread"] = cds["Spread"] / 10000
    cds["Survival Prob"] = 0.0
    cds["Cumulative Hazard"] = 0.0
    cds["Hazard Rate"] = 0.0
    cds = bootstrap_algo(cds)
    print("CDS below")
    print(cds)
    # Plotting CDS spreads with hazard rates together
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.step(cds["Maturity"], cds["Hazard Rate"], marker="*", label="Default Intensity")
    ax2.plot(cds["Maturity"], cds["Spread"], marker="o", color="Blue", label="CDS Spread")
    ax3.plot(cds["Maturity"], cds["Survival Prob"], marker="o", color="Green", label="Survival Probability")

    ax1.set_xlabel("Maturity")
    ax2.set_xlabel("Maturity")
    ax3.set_xlabel("Maturity")
    ax1.set_ylabel("Default Intensity(Hazard Rate)")
    ax2.set_ylabel("CDS Spreads")
    ax3.set_ylabel("Survival Probability")
    fig.suptitle("Term Structure of Hazard Rate / Survival Prob via CDS Spread Curve")
    for row in range(len(cds["Maturity"])):
        ax1.annotate(str(cds["Hazard Rate"][row]), xy=(cds["Maturity"][row], cds["Hazard Rate"][row]))
        ax2.annotate(str(cds["Spread"][row]), xy=(cds["Maturity"][row], cds["Spread"][row]))
        ax3.annotate(str(np.round(cds["Survival Prob"][row], 3)), xy=(cds["Maturity"][row], cds["Survival Prob"][row]))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    print("*************************************************************************")
    print("Simulating Homogeneous Poisson Processes with Derived Hazard Rates...")
    for row in range(1, len(cds)):
        PP1 = Poisson_Process(cds)
        PP1.homogeneous_simulation(cds["Hazard Rate"][row], 1000, 1)
    plt.xlabel("Steps")
    plt.ylabel("Default Occurrences")
    plt.title("Homogeneous Poisson Process of different Hazard Rates")
    plt.legend()
    plt.show()
    print(
        "**********************Be careful running this, it will take 15 minutes for 30000 simulations********************************")
    print("Plotting Poisson Distributions by simulation...")
    all_val = []
    sim_steps = 1000
    for row in range(1, len(cds), 2):
        PP2 = Poisson_Process(cds)
        if row == 1:
            np.random.seed(25)
            all_val.append(PP2.poisson_pdf(cds["Hazard Rate"][row], sim_steps, 30000))
        elif row == 5:
            np.random.seed(24)
            all_val.append(PP2.poisson_pdf(cds["Hazard Rate"][row], sim_steps, 30000))
        else:
            np.random.seed(36)
            all_val.append(PP2.poisson_pdf(cds["Hazard Rate"][row], sim_steps, 30000))

    fig, axs = plt.subplots(1, 3)
    for j in range(3):
        cur_mean = np.mean(all_val[j])
        cur_var = np.var(all_val[j])
        cur_lambd = cds["Hazard Rate"][1 + 2 * j]
        if j == 0:
            sns.distplot(all_val[j], ax=axs[j], bins=35)
        elif j == 2:
            sns.distplot(all_val[j], ax=axs[j], bins=35)
        else:
            sns.distplot(all_val[j], ax=axs[j], bins=35)

        axs[j].axvline(x=cur_var, label="Sim_Var=" + str(np.round(cur_var, 2)), color='b', linestyle='dashed')
        axs[j].axvline(x=cur_mean, label="Sim_Mean=" + str(np.round(cur_mean, 2)), color='yellow', linestyle='dashed')
        axs[j].axvline(x=sim_steps * cur_lambd,
                       label="Theoretical_Mean & Variance=" + str(np.round(sim_steps * cur_lambd, 2)), color='cyan')
        axs[j].set_title("\u03BB = " + str(cur_lambd), fontsize=10)
        if j == 0:
            axs[j].set(ylabel="Probability")
        axs[j].set(xlabel="Event Occurrences")
        axs[j].legend()
    plt.suptitle("Poisson Distribution of Jump Occurrences under different \u03BB(30000 Simulations)")
    plt.show()
    print(
        "************************Be careful running this, it will take 15 minutes for 30000 simulations*************************************************")

    print("Comparing two intervals within one lambda simulation...")
    val1 = []
    val2 = []
    np.random.seed(0)
    sim_steps = 20
    for row in range(1, 4):
        PP3 = Poisson_Process(cds)
        sim_1, sim_2 = PP3.compare_interval(cds["Hazard Rate"][row], sim_steps, 10000)
        val1.append(sim_1)
        val2.append(sim_2)
    fig, axis = plt.subplots(1, 3)
    for j in range(3):
        cur_lambd = np.round(cds["Hazard Rate"][1 + j], 3)
        sns.distplot(val1[j], ax=axis[j], label="Steps-Interval[0:700]")
        sns.distplot(val2[j], ax=axis[j], label="Steps-Interval[300:1000]")
        axis[j].set_title("\u03BB=" + str(cur_lambd), fontsize=10)
        axis[j].set(xlabel="Event Occurrences", ylabel="Probability")
        axis[j].legend()
    plt.suptitle("Jumps Distribution Validation between Different Intervals")
    plt.show()
    print("*************************************************************************")
    # print("Generating a normal random variable with correlation to another normal rvs")
    # RG = Random_Generator("norm")
    # print(RG.Cholesky_Decompose(0.3))
    print("****************************Inhomogeneous Poisson Process Part*********************************************")
    PP4 = Poisson_Process(cds)
    # t = np.linspace(0,10,201)
    # hazard = []
    # for dt in t:
    #     hazard.append(PP4.lambda_t(dt))
    # plt.step(t,hazard,label = "Simulated Hazards")
    # plt.xlabel("Maturity")
    # plt.ylabel("Lambda")
    # plt.title("Hazard Rate Function \u03BB(t)")
    # plt.legend()
    # plt.show()

    # for i in range(1):
    #     Nt, s = PP4.inhomogeneous_simulation(10)
    #     plt.step(s, Nt)
    # plt.xlabel("Days")
    # plt.ylabel("Number of Occurrences")
    # plt.title("Inhomogeneous Poisson Process via implied \u03BB(t)")
    # plt.show()
    val = []
    for i in range(1000):
        val.append(PP4.inhomogeneous_validation(10))
    print("Mean = {}".format(np.mean(val)))
    print("Theoretical = {}".format(0.031 * 2 + 0.029 * 3))
    sns.distplot(val)
    plt.show()
    print("*************************************************************************")
    print("*************************************************************************")
# ****************************************************************#
# Define self-input CDS_Curve
# print("CDS_Curve Generator:")
# term = list(input("Tenor Interval(list): "))
# DF = list(input("Discount Factor(list): "))
# spreads = list(input("CDS Spreads bps(list): "))
# RR = float(input("Recovery Rate: "))
# C = CDS_Curve([0,1,2,3,4,5,7,10],[0,0.97,0.94,0.92,0.89,0.86,0.83,0.79],[0,3,9,15,21,28,43,61],0.4)
# C1 = hazard_rate(C.df)
# print(C1)
