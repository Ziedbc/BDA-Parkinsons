% Chaudhuri, S.E., et al. (2020) Use of Bayesian Decision Analysis to 
% Maximize Value in Patient-Centered Randomized Clinical Trials in 
% Parkinson's Disease

% Copyright (C) 2020 Chaudhuri, S.E., et al.

% This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% For detailed information on GNU General Public License, please visit https://www.gnu.org/licenses/licenses.en.html

clear;
close all;
clc;

% RCT parameters
sigma_c  = 2;       % response variability of primary endpoint metric in control arm (in movement scale units)
sigma_t  = 2;       % response variability of primary endpoint metric in investigational arm (in movement scale units)
eta      = 200;     % accrual rate of new patients per year
s        = 0.5;     % start up time (in years)
f        = 1;       % observation time (in years)
tau      = 0.75;    % FDA review time
maxPower = 0.9;     % contraint on power
     
% social welfare function age-group weights
c.agecat = [1/4,1/4,1/4,1/4];                 % equal-weighted
% c.agecat = [1,0,0,0];                         % age < 61
% c.agecat = [0,1,0,0];                         % 62 <= age < 67
% c.agecat = [0,0,1,0];                         % 67 <= age < 72
% c.agecat = [0,0,0,1];                         % age >= 72 
if sum(c.agecat) ~= 1
    error('Social welfare function weights must sum to 1!');
end

iter = 0;
for c1 = 0:1
for c2 = 0:1 
for c3 = 0:1 
for c4 = 0:1 
for c5 = 0:1 
iter = iter + 1;
fprintf("\n\nCohort %d of %d:\n", iter, 32);

% covariates
c.dbs    = c1; % dbs
c.nonamb = c2; % non-ambulatory
c.cog    = c3; % cognitive symptom
c.mot    = c4; % motor symptoms >= 2
c.dys    = c5; % dyskinesia

% discount rate
R = discount_rate(c);

% control mean difference from baseline (Weaver et al., 2009)
ctrl.ontime     = 0.1;          % benefits
ctrl.movement   = 10 * 1.7/108;
ctrl.pain       = 10 *-1.1/100;
ctrl.cognition  = 10 *-1.6/100;
ctrl.depression = 0;            % risks
ctrl.brainbleed = 0;
ctrl.death      = 0;

% device mean difference from baseline 
dev.ontime     = 4.6;           % hours (Weaver et al., 2009)
dev.movement   = 10 * 12.3/108; % scale to 10 (Weaver et al., 2009)
dev.pain       = 10 * 7.2/100;  % scale to 10 (Weaver et al., 2009)
dev.cognition  = 10 * 3.7/100;  % scale to 10 (Weaver et al., 2009)
dev.depression = 3;             % depression risk in percent (Appleby et al., 2007)      
dev.brainbleed = 1.6;           % intracerebral hemorrhage risk in percent (Kimmelman et al. 2011)
dev.death      = 0.8;           % mortality risk in percent (Weaver et al., 2009)

% device minus control
net.ontime     = dev.ontime     - ctrl.ontime;     % net benefits
net.movement   = dev.movement   - ctrl.movement;
net.pain       = dev.pain       - ctrl.pain;
net.cognition  = dev.cognition  - ctrl.cognition;
net.depression = dev.depression - ctrl.depression; % net risks
net.brainbleed = dev.brainbleed - ctrl.brainbleed;
net.death      = dev.death      - ctrl.death;

% aggregate the net benefits of the device under H = 1
net.benefit  = aggregate_benefits(net,c);

% aggregate the net risks 
net.risk = aggregate_risks(net,c);

% calculate the non-discounted cost of each error
L0 = net.risk;               % false positive (i.e., type 1 error)
L1 = net.benefit - net.risk; % false negative (i.e., type 2 error)

% calculate device treament effect under H = 1 (with respect to movement endpoint)
delta = dev.movement - ctrl.movement;

% set optimizaton search grid
n = 2:1:1000; % subjects per trial arm

% set prior probabilities
p_H0 = 0.5;      % ineffective
p_H1 = 1 - p_H0; % effective

% miscellaneous calculations
I  = sqrt(n/(sigma_c^2+sigma_t^2)); % square root of info per trial
T  = s + 2*n/eta + f + tau;         % trial length (in months) 
DF = exp(-R * T);                   % discount factor due to trial length

% calculate optimal trial designs
[n_star, lambda_star, alpha_star, beta_star] = ...
    optimize_trial(n, I, DF, delta, L0, L1, p_H0, p_H1, maxPower);

% record results
results.dbs{iter,1} = int2yn(c1);
results.nonamb{iter,1} = int2yn(c2);
results.cog{iter,1} = int2yn(c3);
results.mot{iter,1} = int2yn(c4);
results.dys{iter,1} = int2yn(c5);
results.Severity_ratio(iter,1) = L1/L0;
results.Discount_rate(iter,1)  = R;             
results.Sample_size(iter,1)    = 2*n_star;
% results.Critical_value(iter,1) = lambda_star;
results.Significance(iter,1)   = alpha_star;    
results.Power(iter,1)          = 1-beta_star; 

end
end
end
end
end

results_table  = struct2table(results);
results_sorted = sortrows(results_table,'Severity_ratio','ascend');

% display results
fprintf('\n');
disp(results_sorted);

function str = int2yn(x)
    switch x
        case 0
            str = 'No';
        case 1
            str = 'Yes';
        otherwise
            error('Expected 0 or 1!')
    end
end

function r = discount_rate(c) % calculate discount rate 
    r_inv = (6.41 - 0.00) * c.agecat(1) + ...
            (6.41 - 0.19) * c.agecat(2) + ...
            (6.41 - 0.90) * c.agecat(3) + ...
            (6.41 - 1.34) * c.agecat(4);          
    r_inv = r_inv - 0.71*c.nonamb - 0.14*c.cog - 0.69*c.dbs - 0.47*c.dys + 0.47*c.mot;       
    r = 1/r_inv; 
end

function y = aggregate_benefits(x,c) % aggregate attribute benefits in units of mortality risk
    y = ontime_benefit(x,c) + ...
        movement_benefit(x,c) + ...
        pain_benefit(x,c) + ...
        cognition_benefit(x,c);
end

function y = ontime_benefit(x,c) % percentage increase in mortality risk per hour increase in on-time
    y = (0.18 + 0.00) * x.ontime * c.agecat(1) + ...
        (0.18 + 0.11) * x.ontime * c.agecat(2) + ...
        (0.18 + 0.16) * x.ontime * c.agecat(3) + ...
        (0.18 + 0.13) * x.ontime * c.agecat(4);  
    y = y - 0.08*c.nonamb + 0.27*c.cog + 0.45*c.dbs + 0.43*c.dys + 0.3*c.mot;
end

function y = movement_benefit(x,c) % percentage increase in mortality risk per unit decrease in movement scale
    y = (0.36 + 0.00) * x.movement * c.agecat(1) + ...
        (0.36 + 0.15) * x.movement * c.agecat(2) + ...
        (0.36 + 0.03) * x.movement * c.agecat(3) + ...
        (0.36 + 0.07) * x.movement * c.agecat(4);
    y = y + 0.47*c.nonamb + 0.27*c.cog + 0.61*c.dbs - 0.04*c.dys + 0*c.mot;
end

function y = pain_benefit(x,c) % percentage increase in mortality risk per unit decrease in pain scale
    y = (0.35 + 0.00) * x.pain * c.agecat(1) + ...
        (0.35 + 0.09) * x.pain * c.agecat(2) + ...
        (0.35 + 0.06) * x.pain * c.agecat(3) + ...
        (0.35 - 0.02) * x.pain * c.agecat(4);
    y = y + 0.11*c.nonamb + 0.32*c.cog + 0.46*c.dbs - 0.12*c.dys - 0.02*c.mot;
end

function y = cognition_benefit(x,c) % percentage increase in mortality risk per unit decrease in cognition scale
    y = (0.39 + 0.00) * x.cognition * c.agecat(1) + ...
        (0.39 + 0.14) * x.cognition * c.agecat(2) + ...
        (0.39 + 0.20) * x.cognition * c.agecat(3) + ...
        (0.39 + 0.16) * x.cognition * c.agecat(4);
    y = y + 0.23*c.nonamb + 0*c.cog + 0.57*c.dbs - 0.22*c.dys + 0.19*c.mot;
end

function y = aggregate_risks(x,c) % aggregate attribute risks in units of mortality risk
    y = depression_to_death(c) * x.depression + ...
        brainbleed_to_death(c) * x.brainbleed + ...
        1.000 * x.death;
end

function a = depression_to_death(c)
    
    n.ontime     = (0.18 + 0.00) * c.agecat(1) + ... % numerator
                   (0.18 + 0.11) * c.agecat(2) + ...
                   (0.18 + 0.16) * c.agecat(3) + ...
                   (0.18 + 0.13) * c.agecat(4);

    n.movement   = (0.36 + 0.00) * c.agecat(1) + ...
                   (0.36 + 0.15) * c.agecat(2) + ...
                   (0.36 + 0.03) * c.agecat(3) + ...
                   (0.36 + 0.07) * c.agecat(4);

    n.pain       = (0.35 + 0.00) * c.agecat(1) + ...
                   (0.35 + 0.09) * c.agecat(2) + ...
                   (0.35 + 0.06) * c.agecat(3) + ...
                   (0.35 - 0.02) * c.agecat(4);

    n.cognition  = (0.39 + 0.00) * c.agecat(1) + ...
                   (0.39 + 0.14) * c.agecat(2) + ...
                   (0.39 + 0.20) * c.agecat(3) + ...
                   (0.39 + 0.16) * c.agecat(4);
    
    d.ontime     = (2.65 + 0.00) * c.agecat(1) + ... % denominator
                   (2.65 - 0.99) * c.agecat(2) + ...
                   (2.65 - 0.45) * c.agecat(3) + ...
                   (2.65 - 0.99) * c.agecat(4);

    d.movement   = (4.09 + 0.00) * c.agecat(1) + ...
                   (4.09 - 0.49) * c.agecat(2) + ...
                   (4.09 + 0.24) * c.agecat(3) + ...
                   (4.09 - 0.26) * c.agecat(4);

    d.pain       = (3.26 + 0.00) * c.agecat(1) + ...
                   (3.26 - 0.54) * c.agecat(2) + ...
                   (3.26 - 0.08) * c.agecat(3) + ...
                   (3.26 - 1.32) * c.agecat(4);

    d.cognition  = (3.87 + 0.00) * c.agecat(1) + ...
                   (3.87 - 0.55) * c.agecat(2) + ...
                   (3.87 - 1.23) * c.agecat(3) + ...
                   (3.87 - 1.45) * c.agecat(4);
            
    ratio.ontime     = n.ontime / d.ontime;
    ratio.movement   = n.movement / d.movement;
    ratio.pain       = n.pain / d.pain;
    ratio.cognition  = n.cognition / d.cognition;
        
    a = mean([ratio.ontime,ratio.movement,ratio.pain,ratio.cognition]);

end

function a = brainbleed_to_death(c)

    n.ontime     = (0.18 + 0.00) * c.agecat(1) + ... % numerator
                   (0.18 + 0.11) * c.agecat(2) + ...
                   (0.18 + 0.16) * c.agecat(3) + ...
                   (0.18 + 0.13) * c.agecat(4);

    n.movement   = (0.36 + 0.00) * c.agecat(1) + ...
                   (0.36 + 0.15) * c.agecat(2) + ...
                   (0.36 + 0.03) * c.agecat(3) + ...
                   (0.36 + 0.07) * c.agecat(4);

    n.pain       = (0.35 + 0.00) * c.agecat(1) + ...
                   (0.35 + 0.09) * c.agecat(2) + ...
                   (0.35 + 0.06) * c.agecat(3) + ...
                   (0.35 - 0.02) * c.agecat(4);

    n.cognition  = (0.39 + 0.00) * c.agecat(1) + ...
                   (0.39 + 0.14) * c.agecat(2) + ...
                   (0.39 + 0.20) * c.agecat(3) + ...
                   (0.39 + 0.16) * c.agecat(4);
    
    d.ontime     = (0.47 + 0.00) * c.agecat(1) + ... % denominator
                   (0.47 - 0.08) * c.agecat(2) + ...
                   (0.47 + 0.06) * c.agecat(3) + ...
                   (0.47 - 0.01) * c.agecat(4);

    d.movement   = (0.90 + 0.00) * c.agecat(1) + ...
                   (0.90 - 0.15) * c.agecat(2) + ...
                   (0.90 - 0.16) * c.agecat(3) + ...
                   (0.90 + 0.01) * c.agecat(4);

    d.pain       = (0.56 + 0.00) * c.agecat(1) + ...
                   (0.56 - 0.03) * c.agecat(2) + ...
                   (0.56 + 0.01) * c.agecat(3) + ...
                   (0.56 - 0.20) * c.agecat(4);

    d.cognition  = (1.18 + 0.00) * c.agecat(1) + ...
                   (1.18 + 0.01) * c.agecat(2) + ...
                   (1.18 - 0.06) * c.agecat(3) + ...
                   (1.18 - 0.14) * c.agecat(4);
    
    ratio.ontime     = n.ontime / d.ontime;
    ratio.movement   = n.movement / d.movement;
    ratio.pain       = n.pain / d.pain;
    ratio.cognition  = n.cognition / d.cognition;
        
    a = mean([ratio.ontime,ratio.movement,ratio.pain,ratio.cognition]);

end

function [n_star,lambda_star,alpha_star,beta_star] = optimize_trial(n,I,DF,delta,L0,L1,p_H0,p_H1,maxPower) % grid search

    len_n      = length(n);  % get grid size
    
    eCost_H0    = NaN(len_n,1); % expected cost under H = 0 and H = 1
    eCost_H1    = NaN(len_n,1);
    eCost       = NaN(len_n,1);
    alpha       = NaN(len_n,1); % probability of type 1 and type 2 error
    beta        = NaN(len_n,1);
    lambda      = NaN(len_n,1); % decision boundary
    for i = 1:len_n
        
        m         = p_H0 * L0 / (p_H1 * L1);       % LRT threshold
        fun       = @(x) g(x,delta,n(i),I(i),m);   % optimize using ROC tangent
        lambda(i) = fzero(fun,2);
        
        power = 1 - nctcdf(lambda(i),2*(n(i)-1),delta*I(i)); % calculate optimal power
        if power > maxPower                                  % check power constraint
            lambda(i) = nctinv(1-maxPower,2*(n(i)-1),delta*I(i));
            power = maxPower;
        end
        
        p_rej_H0  = nctcdf(lambda(i),2*(n(i)-1),0*I(i));           % probability device is rejected under H = 0
        p_app_H0  = 1 - p_rej_H0;                                  % probability device is approved under H = 0

        p_rej_H1  = 1 - power;                                     % probability device is rejected under H = 1
        p_app_H1  = power;                                         % probability device is approved under H = 1

        eCost_H0(i) = p_rej_H0 * 0 + p_app_H0 * DF(i) * L0;        % expected cost under H = 0
        eCost_H1(i) = p_rej_H1 * L1 + p_app_H1 * (1 - DF(i)) * L1; % expected cost under H = 1
        eCost(i)    = p_H0 * eCost_H0(i) + p_H1 * eCost_H1(i);     % calculate expected cost
        
        alpha(i) = p_app_H0; % significance level
        beta(i)  = p_rej_H1; % 1 - power
        
        if i > 1 && eCost(i) > eCost(i-1) % if expected cost is rising, then minimum has been found
            fprintf('Minimum found!\n');
            break;
        end

        if ~mod(n(i),10), fprintf('Searching: n = %d of %d\n',n(i),n(end)); end
    end

    % find optimal sample size and alpha
    [~,i_star] = min(eCost);
    n_star          = n(i_star);
    lambda_star     = lambda(i_star);
    alpha_star      = alpha(i_star);
    beta_star       = beta(i_star);

end

function y = g(x,delta,n,I,m) % find where cost curve is tangent to ROC 
    y = nctpdf(x,2*(n-1),delta*I) - m * nctpdf(x,2*(n-1),0*I);
end


