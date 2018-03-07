#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:31 2018

@author: pmirbach
"""

import numpy as np
from numpy import sqrt



def A44(U,t):
    u = U/4
    H = np.array([
        [U,     0,      0,      0,      0,      0],
        [0,     5*u-4*t,      sqrt(2)*u,    -sqrt(2)*u,   -sqrt(2)*u,   u],
        [0,     sqrt(2)*u,    U,              0,              U/2,            sqrt(2)*u],
        [0,     -sqrt(2)*u,   0,              U,              -U/2,           -sqrt(2)*u],
        [0,     -sqrt(2)*u,   U/2,            -U/2,           3*U/2,          -sqrt(2)*u],
        [0,     u,            sqrt(2)*u,    -sqrt(2)*u,   -sqrt(2)*u,   5*u+4*t]
        ])
    return H

def B44(U,t):
    u = U/4
    H = np.array([
            [U,     0,              0,                  0],
            [0,     3*u-4*t,      -sqrt(2)*u,    u],
            [0,     -sqrt(2)*u,   U/2,                sqrt(2)*u],
            [0,     u,            sqrt(2)*u,     3*u+4*t]
            ])
    return H

def C44(U,t):
    u = U/4
    H = np.array([
            [5*u-2*t,     -u,              -u,                  u],
            [-u,     5*u-2*t,      u,    -u],
            [-u,     u,   5*u+2*t,                -u],
            [u,     -u,            -u,     5*u+2*t]
            ])
    return H

def D44(U,t):
    u = U/4
    H = np.array([
            [3*u-2*t,     -u,              -u,                  u],
            [-u,          3*u-2*t,         -u,                  u],
            [-u,     -u,              3*u+2*t,                  u],
            [u,     u,              u,                  3*u+2*t],
            ])
    return H

def E44(U,t):
    u = U/4
    H = np.array([
        [3*u-4*t,     sqrt(6)*u,      u,      0,      0,      0],
        [sqrt(6)*u,    3*U/2,      -sqrt(6)*u,    0,   0,   0],
        [u,     -sqrt(6)*u,    3*u+4*t,              0,              0,            0],
        [0,     0,   0,              5*u-4*t,              -sqrt(2)*u,           u],
        [0,     0,   0,            -sqrt(2)*u,           3*U/2,          -sqrt(2)*u],
        [0,     0,   0,             u,   -sqrt(2)*u,   5*u+4*t]
        ])
    return H


def E44_exc(U,t):
    u = U/4
    H = np.array([
        [3*u-4*t,   u],
        [u,     3*u+4*t]
        ])
    return H



def A44_eff(U,Ue,t):
    u = U/4
    ue = Ue/4
    H = np.array([
        [U,     0,      0,      0,      0,      0],
        [0,     5*u-4*t,      sqrt(2)*ue,    -sqrt(2)*ue,   -sqrt(2)*ue,   ue],
        [0,     sqrt(2)*ue,    U,              0,              Ue/2,            sqrt(2)*ue],
        [0,     -sqrt(2)*ue,   0,              U,              -Ue/2,           -sqrt(2)*ue],
        [0,     -sqrt(2)*ue,   Ue/2,            -Ue/2,           3*U/2,          -sqrt(2)*ue],
        [0,     ue,            sqrt(2)*ue,    -sqrt(2)*ue,   -sqrt(2)*ue,   5*u+4*t]
        ])
    return H


def E44_eff(U, Ue, t):
    u = U/4
    ue = Ue/4
    H = np.array([
        [3*u-4*t,       sqrt(6)*ue,      ue,                  0, 0, 0],
        [sqrt(6)*ue,     3*U/2,          -sqrt(6)*ue,         0, 0, 0],
        [ue,             -sqrt(6)*ue,     3*u+4*t,            0, 0, 0],
        [0, 0, 0,       5*u-4*t,        -sqrt(2)*ue,           ue],
        [0, 0, 0,       -sqrt(2)*ue,     3*U/2,          -sqrt(2)*ue],
        [0, 0, 0,       ue,              -sqrt(2)*ue,     5*u+4*t]
        ])
    return H