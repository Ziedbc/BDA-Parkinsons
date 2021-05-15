# Chaudhuri, S., et al. (2021)
# Use of Bayesian Decision Analysis to Maximize Value in Patient-Centered
# Randomized Clinical Trials in Parkinson's Disease,
# Journal: TBD

# Copyright (C) 2021 Chaudhuri, S., et al.

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# For detailed information on GNU General Public License,
# please visit https://www.gnu.org/licenses/licenses.en.html
from typing import List

import pandas as pd
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from scipy.stats import nct

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource, DataTable, TableColumn, Select, BasicTicker, \
    ColorBar, LinearColorMapper, CategoricalColorMapper, PrintfTickFormatter, CheckboxGroup, Button, Legend
from bokeh.models.widgets import Div
from bokeh.plotting import figure
from bokeh.palettes import RdBu3

# Default Values
maxpower_0 = 0.9
p0_0 = 0.5
sigma_c0 = 2  # Response variability of primary endpoint metric in control arm (in movement scale units)
sigma_t0 = 2  # Response variability of primary endpoint metric in investigational arm (in movement scale units)

eta_0 = 200  # Patient Accrual Rate: 100 patients per year
s_0 = 6  # Fixed study start-up time (in months)
f_0 = 12  # Final observation period (in months)
tau_0 = 9  # FDA review time (in months)

# control mean difference from baseline (Weaver et al., 2009)
# benefits
ctrl_ontime_0 = 0.1
ctrl_movement_0 = 10 * 1.7 / 108
ctrl_pain_0 = 10 * (-1.1 / 100)
ctrl_cognition_0 = 10 * (-1.6 / 100)
# risks
ctrl_depression_0 = 0
ctrl_brainbleed_0 = 0
ctrl_death_0 = 0

# device mean difference from baseline
# benefits
dev_ontime_0 = 4.6  # hours (Weaver et al., 2009)
dev_movement_0 = 10 * 12.3 / 108  # scale to 10 (Weaver et al., 2009)
dev_pain_0 = 10 * 7.2 / 100  # scale to 10 (Weaver et al., 2009)
dev_cognition_0 = 10 * 3.7 / 100  # scale to 10 (Weaver et al., 2009)
# risks
dev_depression_0 = 3 / 100  # depression risk in percent (Appleby et al., 2007)
dev_brainbleed_0 = 1.6 / 100  # intracerebral hemorrhage risk in percent (Kimmelman et al. 2011)
dev_death_0 = 0.8 / 100  # mortality risk in percent (Weaver et al., 2009)

# device minus control
# net benefits
net_ontime_0 = dev_ontime_0 - ctrl_ontime_0
net_movement_0 = dev_movement_0 - ctrl_movement_0
net_pain_0 = dev_pain_0 - ctrl_pain_0
net_cognition_0 = dev_cognition_0 - ctrl_cognition_0
# net risks
net_depression_0 = dev_depression_0 - ctrl_depression_0
net_brainbleed_0 = dev_brainbleed_0 - ctrl_brainbleed_0
net_death_0 = dev_death_0 - ctrl_death_0

# Set up widgets
Disease_Name_gui = Select(title="Disease Considered", value="Parkinson's Disease", options=["Parkinson's Disease"])

max_power_check_gui = Select(title="Add Power Constraint: (No = 100% Max Power)", value="Yes", options=["Yes", "No"])
maxPower_gui = Slider(title="Maximal Power Allowed", value=maxpower_0, start=0.0, end=1.0, step=0.01, format="0.0%")
if max_power_check_gui.value == "No":
    maxPower_gui.value = 1.0

p_H0_gui = Slider(title="Probability of an Ineffective Treatment (p\u2080)", value=p0_0, start=0.0, end=1.0, step=0.01,
                  format="0.0%")
sigma_gui = Slider(title="Motor symptom score standard deviation (\u03C3_t and \u03C3_p, out of 10)",
                   value=sigma_c0, start=0.0, end=10.0, step=0.01, format="0[.]00")
net_movement_gui = Slider(title="Motor symptom score mean difference (\u03B4, out of 10)",
                          value=net_movement_0, start=0.0, end=10.0, step=0.01, format="0[.]00")
net_ontime_gui = Slider(title="On-time mean difference (hours, out of 10)",
                        value=net_ontime_0, start=0.0, end=10.0, step=0.01, format="0[.]00")
net_pain_gui = Slider(title="Reduction in pain mean difference (out of 10)",
                      value=net_pain_0, start=0.0, end=10.0, step=0.01, format="0[.]00")
net_cognition_gui = Slider(title="Cognitive symptom mean difference (out of 10)",
                           value=net_cognition_0, start=0.0, end=10.0, step=0.01, format="0[.]00")
net_depression_gui = Slider(title="Depression risk (in %)",
                            value=net_depression_0, start=0.0, end=1.0, step=0.01, format="0[.]00%")
net_brainbleed_gui = Slider(title="Brain-bleed risk (in %)",
                            value=net_brainbleed_0, start=0.0, end=1.0, step=0.01, format="0[.]00%")
net_death_gui = Slider(title="Mortality risk (in %)",
                       value=net_death_0, start=0.0, end=1.0, step=0.01, format="0[.]00%")

eta_gui = Slider(title="Patient Accrual Rate (\u03B7, Patient per Year)",
                 value=eta_0, start=1, end=300, step=1, format="0[.]0")
s_gui = Slider(title="Start-up time before patient enrollment (s, in Months)",
               value=s_0, start=0.0, end=36.0, step=1.0, format="0")
f_gui = Slider(title="Follow-up period after enrolling the last patient (f, in Months)",
               value=f_0, start=0.0, end=48.0, step=1.0, format="0")
tau_gui = Slider(title="FDA Review Time (\u03C4, in Months)",
                 value=tau_0, start=0.0, end=36.0, step=1.0, format="0")

# agecat1_gui ="Proportion of patients 60 years old or younger (in %)"
agecat2_gui = Slider(title="Proportion of patients between 61 and 66 years old (in %)",
                     value=0.25, start=0.0, end=1.0, step=0.01, format="0[.]00%")
agecat3_gui = Slider(title="Proportion of patients between 67 and 71 years old (in %)",
                     value=0.25, start=0.0, end=1.0, step=0.01, format="0[.]00%")
agecat4_gui = Slider(title="Proportion of patients 72 years old or older (in %)",
                     value=0.25, start=0.0, end=1.0, step=0.01, format="0[.]00%")

submit = Button(label='Calculate', button_type='success', visible=True)

# Covariates
covariates_Label = ["Non-ambulatory (If patients report problems with balance or walking)",
                    "Cognitive symptom (If patients report difficulty thinking clearly)",
                    "DBS (If patients report received Deep Brain Stimulation)",
                    "Dyskinesia (If patients experience dyskinesia as a side effect of their medication)",
                    "Motor symptom (If patients have impaired motor function)"]
covariates_gui = CheckboxGroup(labels=covariates_Label, active=[0, 1])
covariates_gui.active = []

# Helper: get covariates vector from covariates_gui
# Encode covariates in term of 0/1:
def get_covariates_vect(covs):
    """
    nonamb = covariates_list[0]
    cog = covariates_list[1]
    dbs = covariates_list[2]
    dys = covariates_list[3]
    mot = covariates_list[4]
    """
    covariates_list = [0] * len(covs.labels)
    for i in range(len(covs.labels)):
        if i in covs.active:
            covariates_list[i] = 1
    return covariates_list


covariates_vec = get_covariates_vect(covariates_gui)


# BDA Calculations:

def Run_BDA(
        Disease_Name, maxPower, p_H0, sigma,
        net_movement, net_ontime, net_pain, net_cognition,
        net_depression, net_brainbleed, net_death,
        eta, s, f, tau, agecat2, agecat3, agecat4, covariates):
    # Miscellaneous calculations
    p_H1 = 1 - p_H0
    agecat1 = 1 - agecat2 - agecat3 - agecat4
    # Convert timeline from months to years
    s = s / 12
    f = f / 12
    tau = tau / 12
    # Convert from decimal to %:
    net_depression = 100 * net_depression
    net_brainbleed = 100 * net_brainbleed
    net_death = 100 * net_death

    # Encode covariates in term of 0/1:
    covariates_list = get_covariates_vect(covariates)
    nonamb = covariates_list[0]
    cog = covariates_list[1]
    dbs = covariates_list[2]
    dys = covariates_list[3]
    mot = covariates_list[4]

    # Helpers:
    def I_n(n, sigma_c=sigma, sigma_t=sigma):  # square root of info per trial
        I = (n / (sigma_c ** 2 + sigma_t ** 2)) ** 0.5
        return I

    def T_n(n, s=s, eta=eta, f=f, tau=tau):  # trial length (in years)
        T = s + 2 * n / eta + f + tau
        return T

    # Get Discount rate
    def get_R(agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4,
              nonamb=nonamb, cog=cog, dbs=dbs, dys=dys, mot=mot):
        r_inv = (6.41 - 0.00) * agecat1 + \
                (6.41 - 0.19) * agecat2 + \
                (6.41 - 0.90) * agecat3 + \
                (6.41 - 1.34) * agecat4
        r_inv = r_inv - 0.71 * nonamb - 0.14 * cog - 0.69 * dbs - 0.47 * dys + 0.47 * mot
        r = 1 / r_inv
        return r

    # discount factor due to trial length
    def DF_Rn(n):
        R = get_R()
        T = T_n(n)
        DF = np.exp(-R * T)
        return DF

    # aggregate attribute benefits in units of mortality risk
    def aggregate_benefits():
        y = ontime_benefit() + movement_benefit() + \
            pain_benefit() + cognition_benefit()
        return y

    # percentage increase in mortality risk per hour increase in on-time
    def ontime_benefit(net_ontime=net_ontime,
                       agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4,
                       nonamb=nonamb, cog=cog, dbs=dbs, dys=dys, mot=mot):
        y = (0.18 + 0.00) * net_ontime * agecat1 + \
            (0.18 + 0.11) * net_ontime * agecat2 + \
            (0.18 + 0.16) * net_ontime * agecat3 + \
            (0.18 + 0.13) * net_ontime * agecat4
        y = y - 0.08 * nonamb + 0.27 * cog + 0.45 * dbs + 0.43 * dys + 0.3 * mot
        return y

    # percentage increase in mortality risk per unit decrease in movement scale
    def movement_benefit(net_movement=net_movement,
                         agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4,
                         nonamb=nonamb, cog=cog, dbs=dbs, dys=dys, mot=mot):
        y = (0.36 + 0.00) * net_movement * agecat1 + \
            (0.36 + 0.15) * net_movement * agecat2 + \
            (0.36 + 0.03) * net_movement * agecat3 + \
            (0.36 + 0.07) * net_movement * agecat4
        y = y + 0.47 * nonamb + 0.27 * cog + 0.61 * dbs - 0.04 * dys + 0 * mot
        return y

    # percentage increase in mortality risk per unit decrease in pain scale
    def pain_benefit(net_pain=net_pain,
                     agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4,
                     nonamb=nonamb, cog=cog, dbs=dbs, dys=dys, mot=mot):
        y = (0.35 + 0.00) * net_pain * agecat1 + \
            (0.35 + 0.09) * net_pain * agecat2 + \
            (0.35 + 0.06) * net_pain * agecat3 + \
            (0.35 - 0.02) * net_pain * agecat4
        y = y + 0.11 * nonamb + 0.32 * cog + 0.46 * dbs - 0.12 * dys - 0.02 * mot
        return y

    # percentage increase in mortality risk per unit decrease in cognition scale
    def cognition_benefit(net_cognition=net_cognition,
                          agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4,
                          nonamb=nonamb, cog=cog, dbs=dbs, dys=dys, mot=mot):
        y = (0.39 + 0.00) * net_cognition * agecat1 + \
            (0.39 + 0.14) * net_cognition * agecat2 + \
            (0.39 + 0.20) * net_cognition * agecat3 + \
            (0.39 + 0.16) * net_cognition * agecat4
        y = y + 0.23 * nonamb + 0 * cog + 0.57 * dbs - 0.22 * dys + 0.19 * mot
        return y

    # aggregate attribute risks in units of mortality risk
    def aggregate_risks(net_depression=net_depression, net_brainbleed=net_brainbleed, net_death=net_death):
        y = depression_to_death() * net_depression + \
            brainbleed_to_death() * net_brainbleed + \
            1.000 * net_death
        return y

    def depression_to_death(agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4):
        # numerator
        n_ontime = (0.18 + 0.00) * agecat1 + \
                   (0.18 + 0.11) * agecat2 + \
                   (0.18 + 0.16) * agecat3 + \
                   (0.18 + 0.13) * agecat4

        n_movement = (0.36 + 0.00) * agecat1 + \
                     (0.36 + 0.15) * agecat2 + \
                     (0.36 + 0.03) * agecat3 + \
                     (0.36 + 0.07) * agecat4

        n_pain = (0.35 + 0.00) * agecat1 + \
                 (0.35 + 0.09) * agecat2 + \
                 (0.35 + 0.06) * agecat3 + \
                 (0.35 - 0.02) * agecat4

        n_cognition = (0.39 + 0.00) * agecat1 + \
                      (0.39 + 0.14) * agecat2 + \
                      (0.39 + 0.20) * agecat3 + \
                      (0.39 + 0.16) * agecat4

        # denominator
        d_ontime = (2.65 + 0.00) * agecat1 + \
                   (2.65 - 0.99) * agecat2 + \
                   (2.65 - 0.45) * agecat3 + \
                   (2.65 - 0.99) * agecat4

        d_movement = (4.09 + 0.00) * agecat1 + \
                     (4.09 - 0.49) * agecat2 + \
                     (4.09 + 0.24) * agecat3 + \
                     (4.09 - 0.26) * agecat4

        d_pain = (3.26 + 0.00) * agecat1 + \
                 (3.26 - 0.54) * agecat2 + \
                 (3.26 - 0.08) * agecat3 + \
                 (3.26 - 1.32) * agecat4

        d_cognition = (3.87 + 0.00) * agecat1 + \
                      (3.87 - 0.55) * agecat2 + \
                      (3.87 - 1.23) * agecat3 + \
                      (3.87 - 1.45) * agecat4

        ratio_ontime = n_ontime / d_ontime
        ratio_movement = n_movement / d_movement
        ratio_pain = n_pain / d_pain
        ratio_cognition = n_cognition / d_cognition

        a = np.mean([ratio_ontime, ratio_movement, ratio_pain, ratio_cognition])
        return a

    def brainbleed_to_death(agecat1=agecat1, agecat2=agecat2, agecat3=agecat3, agecat4=agecat4):
        # numerator
        n_ontime = (0.18 + 0.00) * agecat1 + \
                   (0.18 + 0.11) * agecat2 + \
                   (0.18 + 0.16) * agecat3 + \
                   (0.18 + 0.13) * agecat4

        n_movement = (0.36 + 0.00) * agecat1 + \
                     (0.36 + 0.15) * agecat2 + \
                     (0.36 + 0.03) * agecat3 + \
                     (0.36 + 0.07) * agecat4

        n_pain = (0.35 + 0.00) * agecat1 + \
                 (0.35 + 0.09) * agecat2 + \
                 (0.35 + 0.06) * agecat3 + \
                 (0.35 - 0.02) * agecat4

        n_cognition = (0.39 + 0.00) * agecat1 + \
                      (0.39 + 0.14) * agecat2 + \
                      (0.39 + 0.20) * agecat3 + \
                      (0.39 + 0.16) * agecat4

        # denominator
        d_ontime = (0.47 + 0.00) * agecat1 + \
                   (0.47 - 0.08) * agecat2 + \
                   (0.47 + 0.06) * agecat3 + \
                   (0.47 - 0.01) * agecat4

        d_movement = (0.90 + 0.00) * agecat1 + \
                     (0.90 - 0.15) * agecat2 + \
                     (0.90 - 0.16) * agecat3 + \
                     (0.90 + 0.01) * agecat4

        d_pain = (0.56 + 0.00) * agecat1 + \
                 (0.56 - 0.03) * agecat2 + \
                 (0.56 + 0.01) * agecat3 + \
                 (0.56 - 0.20) * agecat4

        d_cognition = (1.18 + 0.00) * agecat1 + \
                      (1.18 + 0.01) * agecat2 + \
                      (1.18 - 0.06) * agecat3 + \
                      (1.18 - 0.14) * agecat4

        ratio_ontime = n_ontime / d_ontime
        ratio_movement = n_movement / d_movement
        ratio_pain = n_pain / d_pain
        ratio_cognition = n_cognition / d_cognition

        a = np.mean([ratio_ontime, ratio_movement, ratio_pain, ratio_cognition])
        return a

    # aggregate the net benefits of the device under H = 1
    net_benefit = aggregate_benefits()
    # aggregate the net risks
    net_risk = aggregate_risks()
    # calculate the non-discounted cost of each error
    L0 = net_risk  # false positive (i.e., type 1 error)
    L1 = net_benefit - net_risk  # false negative (i.e., type 2 error)
    # calculate device treatment effect under H = 1 (with respect to movement endpoint)
    delta = net_movement

    # Get Loss Function:
    # L_0: Relative loss in value per person of using the investigational device under the null hypothesis (H = 0)
    # L_1: Relative loss in value per person of forgoing the use of the investigational device under the H = 1
    # Get E[Loss|H_0]:
    def get_cost_h0(n, lambda_n, L0=L0):
        p_rej_H0 = nct.cdf(lambda_n, 2 * (n - 1), 0 * I_n(n))  # probability device is rejected under H = 0
        p_app_H0 = 1 - p_rej_H0  # probability device is approved under H = 0
        eCost_H0 = p_rej_H0 * 0 + p_app_H0 * DF_Rn(n) * L0
        return eCost_H0

    # Get E[Loss|H_1]:
    def get_cost_h1(n, lambda_n, L1=L1):
        p_rej_H1 = 1 - get_Power(lambda_n, n)  # probability device is rejected under H = 0
        p_app_H1 = 1 - p_rej_H1  # probability device is approved under H = 0
        eCost_H1 = p_rej_H1 * L1 + p_app_H1 * (1 - DF_Rn(n)) * L1
        return eCost_H1

    # Expected Cost Function:
    def C_n(targets, p_H0=p_H0):
        # Parameters
        n, lambda_n = targets[0], targets[1]
        # Get E[C|H_0]
        cost_h0 = get_cost_h0(n, lambda_n)
        # Get E[C|H_1]
        cost_h1 = get_cost_h1(n, lambda_n)
        return p_H0 * cost_h0 + (1 - p_H0) * cost_h1

    # Find alpha Given the Critical Value lambda_n:
    def get_alpha(lambda_n, n):
        p_rej_H0 = nct.cdf(lambda_n, 2 * (n - 1), 0 * I_n(n))
        return 1 - p_rej_H0

    # Find Power Given the Critical Value lambda_n:
    def get_Power(lambda_n, n, delta=delta):
        power = 1 - nct.cdf(lambda_n, 2 * (n - 1), delta * I_n(n))
        return power

    # Define the Power Constraint: Power <= maxPower
    def power_constr(targets, maxPower=maxPower):
        return -(maxPower - get_Power(targets[1], targets[0]))

    # Find beta Given the Power: beta = 1 - beta
    def get_beta(Power):
        return 1 - Power

    # Calculate Optimal Sample Size & Critical Value:

    # Unconstrained Minimization:
    bnds = ((1, 500), (0.001, 10))
    # result_unc = optimize.minimize(C_n, np.array([140, 3.0]), bounds=bnds, method='trust-constr')
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
    result_unc = optimize.basinhopping(C_n, np.array([140, 3.0]), minimizer_kwargs=minimizer_kwargs)
    n_unc, lambda_unc = result_unc.x
    cost_unc = result_unc.fun
    alpha_unc, power_unc = get_alpha(lambda_unc, n_unc), get_Power(lambda_unc, n_unc)
    EC1_unc = get_cost_h0(n_unc, lambda_unc)  # E[C|H_0]
    EC2_unc = get_cost_h1(n_unc, lambda_unc)  # E[C|H_1]

    # Constrained Maximization: with Power <= maxPower
    if power_unc <= maxPower:
        n_con, lambda_con = n_unc, lambda_unc
        cost_con = cost_unc
        alpha_con, power_con = alpha_unc, power_unc
        EC1_con = EC1_unc
        EC2_con = EC2_unc
    else:
        nlc = NonlinearConstraint(power_constr, -np.inf, 0)
        # result_con = optimize.minimize(C_n, np.array([n_unc, lambda_unc]), bounds=bnds, method='trust-constr', constraints=nlc)
        minimizer_kwargs2 = {'method': "COBYLA", 'bounds': bnds, "constraints": nlc}
        result_con = optimize.basinhopping(C_n, np.array([n_unc, lambda_unc]), minimizer_kwargs=minimizer_kwargs2)
        n_con, lambda_con = result_con.x
        cost_con = result_con.fun
        alpha_con, power_con = get_alpha(lambda_con, n_con), get_Power(lambda_con, n_con)
        EC1_con = get_cost_h0(n_con, lambda_con)  # E[C|H_0]
        EC2_con = get_cost_h1(n_con, lambda_con)  # E[C|H_1]

    # Collect the Results Into a Data Frame:
    results_df = pd.DataFrame(data=
                              {'Disease Name': [Disease_Name],
                               'L0': [round(L0, 2)],
                               'L1': [round(L1, 2)],
                               'Severity Ratio (L1/L0)': [round(L1/L0, 2)],
                               'Discount Rate (R, in %)': [round(100*get_R(), 1)],
                               'Trial Size (2n)': [int(round(2*n_con, 0))],
                               'Critical Value': [round(lambda_con, 3)],
                               '\u03B1 (%)': [round(100 * alpha_con, 2)],
                               'Power (%)': [round(100 * power_con, 1)]})
    EC0 = [round(cost_con, 3)] #["{:.1e}".format(cost_con)]
    EC1 = [round(EC1_con, 3)] #["{:.1e}".format(EC1_con)]
    EC2 = [round(EC2_con, 3)] #["{:.1e}".format(EC2_con)]

    # Plot Contour Level Lines:
    # Specify Meshgrid Dimensions
    delta_x, delta_y, x_interv = 10, 0.2, 200
    max_n = round(n_con, 0)
    n_x = np.arange(2, max_n + x_interv, delta_x)
    # lambda_y = np.arange(-1.0, 2.0, delta_y)
    lambda_y = np.arange(-0.1, 5.0, delta_y)
    n_X, lambda_Y = np.meshgrid(n_x, lambda_y)

    # Compute Expected Cost Over the Grid:
    Exp_Cost = np.empty((len(lambda_y), len(n_x)))
    for j in range(len(n_x)):
        for i in range(len(lambda_y)):
            Exp_Cost[i, j] = C_n([n_X[i, j], lambda_Y[i, j]])

    dic_plt = pd.DataFrame({'n_X': np.concatenate(n_X).flat, 'lambda_Y': np.concatenate(lambda_Y).flat,
                            'Exp_Cost': np.concatenate(Exp_Cost).flat})
    dic_scatter = dict({'x': [n_unc, n_con], 'y': [lambda_unc, lambda_con], 'label': ['Unconstrained', 'Constrained']})

    return results_df, dic_scatter, dic_plt, EC0, EC1, EC2


# Set up Table Output:
BDA_res, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(
    Disease_Name_gui.value, maxPower_gui.value, p_H0_gui.value, sigma_gui.value,
    net_movement_gui.value, net_ontime_gui.value, net_pain_gui.value, net_cognition_gui.value,
    net_depression_gui.value, net_brainbleed_gui.value, net_death_gui.value, eta_gui.value, s_gui.value, f_gui.value,
    tau_gui.value, agecat2_gui.value, agecat3_gui.value, agecat4_gui.value, covariates_gui)
Columns = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns]  # bokeh columns
Columns1 = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns[0:5]]  # bokeh columns
Columns2 = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns[5:]]  # bokeh columns
Columns3 = [TableColumn(field='Optimal Cost', title='Optimal Cost'),
            TableColumn(field='E[C|H_0]', title='E[C|H\u2080]'),
            TableColumn(field='E[C|H_1]', title='E[C|H\u2081]')]  # bokeh columns

source_df = ColumnDataSource(data=BDA_res)
source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'E[C|H_0]': EC1, 'E[C|H_1]': EC2})

source_obj = DataTable(columns=Columns, source=source_df, width=1000, height=75, index_position=None)  # bokeh table
source_obj1 = DataTable(columns=Columns1, source=source_df, width=700, height=60, index_position=None)  # bokeh table
source_obj2 = DataTable(columns=Columns2, source=source_df, width=400, height=60, index_position=None)  # bokeh table
source_obj3 = DataTable(columns=Columns3, source=source_df3, width=300, height=60, index_position=None)  # bokeh table

# Set up Plot Output:
plot_opt = figure(plot_width=750, plot_height=525, tools='pan, wheel_zoom,box_select,reset, save',
                  x_axis_label='Sample Size n', y_axis_label='Critical Value \u03BB')
plot_opt.title.text = '%s (Contour Plot)' % Disease_Name_gui.value

colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=dic_plt.Exp_Cost.min(), high=dic_plt.Exp_Cost.max())
source_plt = ColumnDataSource(data=dic_plt)
plot_opt.rect(source=source_plt, x='n_X', y='lambda_Y', width=20, height=0.2, line_color=None,
              fill_color={'field': 'Exp_Cost', 'transform': mapper})
color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="12px",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%.1e"),
                     label_standoff=13, border_line_color=None, location=(0, 0))
plot_opt.add_layout(color_bar, 'right')
plot_opt.add_layout(Legend(), 'right')

source_scatter = ColumnDataSource(data=dic_scatter)
cat_color_mapper = CategoricalColorMapper(factors=['Unconstrained', 'Constrained'], palette=[RdBu3[0], RdBu3[2]])
plot_opt.circle('x', 'y', size=20, source=source_scatter, color={'field': 'label', 'transform': cat_color_mapper},
                legend_group='label', selection_color="orange", alpha=0.6,
                nonselection_alpha=0.1, selection_alpha=0.4)

# Spinner: https://www.w3schools.com/howto/howto_css_loader.asp
spinner_text = """
<!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
<div class="loader">
<style scoped>
.loader {
    border: 16px solid #f3f3f3; /* Light grey */
    border-top: 16px solid #3498db; /* Blue */
    border-radius: 50%;
    width: 120px;
    height: 120px;
    animation: spin 2s linear infinite;
    left:0;
    right:0;
    top:0;
    bottom:0;
    position:absolute;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
} 
</style>
</div>
"""
div_spinner = Div(text="", width=0, height=0)


def show_spinner1():
    div_spinner.width = 120
    div_spinner.height = 120
    div_spinner.text = spinner_text


def show_spinner2():
    div_spinner.width = 120
    div_spinner.height = 130
    div_spinner.text = spinner_text
    curdoc().add_next_tick_callback(update_data2)


def hide_spinner():
    div_spinner.width = 0
    div_spinner.height = 0
    div_spinner.text = ""
# End of Spinner


# Set up callbacks
def update_data(attrname, old, new):
    # Get the current slider values
    if (old in ['Yes', 'No']) or (new in ['Yes', 'No']):
        if max_power_check_gui.value == 'No':
            maxPower_gui.value = 1.0

    if maxPower_gui.value < 1.0:
        max_power_check_gui.value = 'Yes'
    else:
        max_power_check_gui.value = 'No'

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(
        Disease_Name_gui.value, maxPower_gui.value, p_H0_gui.value, sigma_gui.value,
        net_movement_gui.value, net_ontime_gui.value, net_pain_gui.value, net_cognition_gui.value,
        net_depression_gui.value, net_brainbleed_gui.value, net_death_gui.value, eta_gui.value, s_gui.value,
        f_gui.value, tau_gui.value, agecat2_gui.value, agecat3_gui.value, agecat4_gui.value, covariates_gui)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'E[C|H_0]': EC1, 'E[C|H_1]': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name_gui.value
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()


def update_data1(attrname, old, new):
    curdoc().add_next_tick_callback(show_spinner1)
    # Get the current slider values
    if (old in ['Yes', 'No']) or (new in ['Yes', 'No']):
        if max_power_check_gui.value == 'No':
            maxPower_gui.value = 1.0

    if maxPower_gui.value < 1.0:
        max_power_check_gui.value = 'Yes'
    else:
        max_power_check_gui.value = 'No'

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(
        Disease_Name_gui.value, maxPower_gui.value, p_H0_gui.value, sigma_gui.value,
        net_movement_gui.value, net_ontime_gui.value, net_pain_gui.value, net_cognition_gui.value,
        net_depression_gui.value, net_brainbleed_gui.value, net_death_gui.value, eta_gui.value, s_gui.value,
        f_gui.value, tau_gui.value, agecat2_gui.value, agecat3_gui.value, agecat4_gui.value, covariates_gui)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'E[C|H_0]': EC1, 'E[C|H_1]': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name_gui.value
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()

    curdoc().add_next_tick_callback(hide_spinner)


def update_data2():
    #curdoc().add_next_tick_callback(show_spinner)
    # Get the current slider values
    if maxPower_gui.value < 1.0:
        max_power_check_gui.value = 'Yes'
    else:
        max_power_check_gui.value = 'No'

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(
        Disease_Name_gui.value, maxPower_gui.value, p_H0_gui.value, sigma_gui.value,
        net_movement_gui.value, net_ontime_gui.value, net_pain_gui.value, net_cognition_gui.value,
        net_depression_gui.value, net_brainbleed_gui.value, net_death_gui.value, eta_gui.value, s_gui.value,
        f_gui.value, tau_gui.value, agecat2_gui.value, agecat3_gui.value, agecat4_gui.value, covariates_gui)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'E[C|H_0]': EC1, 'E[C|H_1]': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name_gui.value
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()

    curdoc().add_next_tick_callback(hide_spinner)


# s_gui.value, f_gui.value, tau_gui.value, eta_gui.value, p_H0_gui.value, N_gui.value, maxPower_gui.value, Disease_Name_gui.value, mort_t_gui.value, mort_p_gui.value, mu_t_gui.value, mu_p_gui.value, r_gui.value, gamma_gui.value, sigma_gui.value
# for params_gui in [max_power_check_gui, maxPower_gui, p_H0_gui, sigma_gui, #Disease_Name_gui,
#                    net_movement_gui, net_ontime_gui, net_pain_gui, net_cognition_gui,
#                    net_depression_gui, net_brainbleed_gui, net_death_gui,
#                    eta_gui, s_gui, f_gui, tau_gui,
#                    agecat2_gui, agecat3_gui, agecat4_gui]:
#     params_gui.on_change('value', update_data)
#     covariates_gui.on_change('active', update_data)

# Update data from boxes
for params_gui in [Disease_Name_gui, max_power_check_gui]:
    params_gui.on_change('value', update_data1)

# Update data from sliders
submit.on_click(show_spinner2)

# Set up Table Layout
Tab1_title = Div(text='<b>Model Inputs:</b>')
Tab2_title = Div(text='<b>BDA Results: (Constrained)</b>')
Tab3_title = Div(text='<b>Optimal Cost: (Constrained)</b>')

# Set up layouts and add to document
sub_widgets = column(column(Tab1_title, source_obj1),
                     row(column(Tab2_title, source_obj2), column(Tab3_title, source_obj3)))
col_sep_0 = column(Div(text=''), width=100)
widgets = column(row(col_sep_0, plot_opt), row(col_sep_0, sub_widgets))


# Input Divider:
#Wid0_title = Div(text='<i>The default input values are taken from Table 3 of the paper.</i>')
Wid_submit = Div(text='<b>Click Here to Calculate the BDA Outputs:</b>')
Wid1_title = Div(text='<b>Disease:</b>')
Wid2_title = Div(text='<b>Power Constraint:</b>')
Wid3_title = Div(text='<b>Device Characteristics:</b>')
Wid4_title = Div(text='<b>Device Benefits:</b>')
Wid5_title = Div(text='<b>Device Risks:</b>')
Wid6_title = Div(text='<b>Timeline Specifications:</b>')
Wid7_title = Div(text='<b>Population Demographics:</b>')
Wid8_title = Div(text='<b>Population Symptoms:</b>')

Wid_separator = Div(text='')
inputs_title = row(Div(text="<h1 style='Black: blue; font-size: 20px'> Model Parameters: </h1>"),
                   Div(text="<i></br>The default input values are taken from Table 3 of the paper.</i>", width=450))
inputs_1 = column(Wid1_title, Disease_Name_gui, Wid2_title, max_power_check_gui, maxPower_gui, width=450)
inputs_2 = column(Wid3_title, p_H0_gui, sigma_gui, width=450)
inputs_3 = column(Wid4_title, net_movement_gui, net_ontime_gui, net_pain_gui, net_cognition_gui, width=450)
inputs_4 = column(Wid5_title, net_depression_gui, net_brainbleed_gui, net_death_gui, width=450)
inputs_5 = column(Wid6_title, eta_gui, s_gui, f_gui, tau_gui, width=450)
inputs_6 = column(Wid7_title, agecat2_gui, agecat3_gui, agecat4_gui, width=450)
inputs_7 = column(Wid8_title, covariates_gui, width=450)
col_sep = column(Div(text=''), width=20)
inputs = column(inputs_title, Wid_submit, submit, div_spinner, row(inputs_1, col_sep, inputs_2),
                Wid_separator, row(inputs_3, col_sep, inputs_4),
                Wid_separator, row(inputs_6, col_sep, inputs_7), Wid_separator, row(inputs_5))

# GUI Title
gui_title = Div(text="<b style='color: DarkRed; font-size: 40px'>Use of Bayesian Decision Analysis to Maximize"
                     "</br>Value in Patient-Centered Randomized Clinical"
                     "</br>Trials in Parkinson's Disease</b> &emsp;&emsp;&emsp;&emsp;" +
                     # "<a href='https://TBD'>"
                     "<i style='color: BlueViolet; font-size: 20px'>(Chaudhuri et al.)</i>", width=1100)
                     # "</a>", width=1100)

# GUI Notes
gui_notes = Div(text="<h1 style='Black: blue; font-size: 18px'> Notes: </h1>" +
                     "<p style='color: DimGrey; font-size: 10px'> <b> (1) Prevalence (N): </b>"
                     "Size of patient population.</p>" +
                     "<p style='color: DimGrey; font-size: 10px'> <b> (2) Incidence Rate (I): </b>"
                     "Rate of increase for new patients. </p>" +
                     "<p style='color: DimGrey; font-size: 10px'> <b> (3) %TBWL (y): </b>"
                     "Total Body Weight Loss, as a Percentage of Initial Weight. </p>")


main_row = column(widgets, Wid_separator, inputs)
main_col = column(gui_title, main_row)  # Can also include: gui_notes

curdoc().add_root(main_col)
curdoc().title = "Maximize Value in Patient-Centered Randomized Clinical Trials in Parkinson's Disease"
