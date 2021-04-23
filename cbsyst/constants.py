"""
Equilibrium speciation constants for carbon and boron in seawater.

%  (***) Each element must be an integer, 
%        indicating the K1 K2 dissociation constants that are to be used:
%   1 = Roy, 1993											T:    0-45  S:  5-45. Total scale. Artificial seawater.
%   2 = Goyet & Poisson										T:   -1-40  S: 10-50. Seaw. scale. Artificial seawater.
%   3 = HANSSON              refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   4 = MEHRBACH             refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   5 = HANSSON and MEHRBACH refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   6 = GEOSECS (i.e., original Mehrbach)					T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   7 = Peng	(i.e., originam Mehrbach but without XXX)	T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   8 = Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)	T:    0-50  S:     0. 
%   9 = Cai and Wang, 1998									T:    2-35  S:  0-49. NBS scale.   Real and artificial seawater.
%  10 = Lueker et al, 2000									T:    2-35  S: 19-43. Total scale. Real seawater.
%  11 = Mojica Prieto and Millero, 2002.					T:    0-45  S:  5-42. Seaw. scale. Real seawater
%  12 = Millero et al, 2002									T: -1.6-35  S: 34-37. Seaw. scale. Field measurements.
%  13 = Millero et al, 2006									T:    0-50  S:  1-50. Seaw. scale. Real seawater.
%  14 = Millero et al, 2010									T:    0-50  S:  1-50. Seaw. scale. Real seawater.
% 
%  (****) Each element must be an integer that 
%         indicates the KSO4 dissociation constants that are to be used,
%         in combination with the formulation of the borate-to-salinity ratio to be used.
%         Having both these choices in a single argument is somewhat awkward, 
%         but it maintains syntax compatibility with the previous version.
%  1 = KSO4 of Dickson & TB of Uppstrom 1979  (PREFERRED) 
%  2 = KSO4 of Khoo    & TB of Uppstrom 1979
%  3 = KSO4 of Dickson & TB of Lee 2010
%  4 = KSO4 of Khoo    & TB of Lee 2010
"""
import numpy as np

# Carbon system


def K0(T, Sal, mode="Weiss1974"):
    opts = ["Weiss1974"]
    if mode == "Weiss1974":
        return np.exp(
            -60.2409
            + 93.4517 * 100 / T
            + 23.3585 * np.log(T / 100)
            + Sal * (0.023517 - 0.023656 * T / 100 + 0.0047036 * (T / 100) * (T / 100))
        )  # Weiss74


def K1K2(T, Sal, P, mode="Luecker2000"):
    opts = ["Luecker2000"]
    if mode == "Luecker2000":
        pass
    return


def KSO4(T, Sal, P, mode="Dickson1990"):
    opts = ["Dickson1990", "Khoo1977"]
    if mode == "Dickson1990":
        par = np.array(
            [
                141.328,
                -4276.1,
                -23.093,
                324.57,
                -13856,
                -47.986,
                -771.54,
                35474,
                114.723,
                -2698,
                1776,
            ]
        )  # Dickson 1990

        return np.exp(
            par[0]
            + par[1] / T
            + par[2] * np.log(T)
            + np.sqrt(Istr) * (par[3] + par[4] / T + par[5] * np.log(T))
            + Istr * (par[6] + par[7] / T + par[8] * np.log(T))
            + par[9] / T * Istr * np.sqrt(Istr)
            + par[10] / T * Istr ** 2
            + np.log(1 - 0.001005 * Sal)
        )

    elif mode == "Khoo 1977":
        pass
