import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C

def MinimosCuadrados(puntosx, puntosy):
    n = len(puntosx)
    graf = []
    valor_x = np.linspace(np.amin(puntosx), np.amax(puntosx), 1000) 
    a1 = (n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy))/(n*sum(np.power(puntosx, 2)) - (sum(puntosx))**2)
    a0 = np.mean(puntosy) - a1*np.mean(puntosx)
    r1 = n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy)
    r2 = np.sqrt(n*sum(np.power(puntosx, 2)) - sum(puntosx)**2)*np.sqrt(n*sum(np.power(puntosy, 2)) - sum(puntosy)**2)
    r = r1/r2
    sigma = np.sqrt(sum((puntosy - a1*puntosx - a0)**2)/(n - 2))
    E_a1 = (np.sqrt(n)*sigma)/np.sqrt(n*sum(np.power(puntosx, 2)) - sum(puntosx)**2)
    for x in valor_x:
        y = a0 + a1*x
        graf.append(y)
    return a0, a1, valor_x, graf, r, E_a1

ValoresT1_1 = np.loadtxt('datos_1.txt', skiprows = 9, usecols = (0, 1, 2, 3, 4, 5), max_rows = 41)
ValoresT1_2 = np.loadtxt('datos_1.txt', skiprows = 56, usecols = (0, 1, 2, 3, 4), max_rows = 21)
ValoresT1_3 = np.loadtxt('datos_1.txt', skiprows = 83, usecols = (0, 1, 2, 3, 4), max_rows = 19)
ValoresT1_4 = np.loadtxt('datos_1.txt', skiprows = 107, usecols = (0, 1, 2), max_rows = 21)

ValoresT2_1 = np.loadtxt('datos_2.txt', skiprows = 9, usecols = (0, 1, 2, 3, 4, 5), max_rows = 41)
ValoresT2_2 = np.loadtxt('datos_2.txt', skiprows = 56, usecols = (0, 1, 2, 3, 4), max_rows = 21)
ValoresT2_3 = np.loadtxt('datos_2.txt', skiprows = 83, usecols = (0, 1, 2, 3, 4), max_rows = 19)
ValoresT2_4 = np.loadtxt('datos_2.txt', skiprows = 107, usecols = (0, 1, 2), max_rows = 21)

ValoresT3_1 = np.loadtxt('datos_3.txt', skiprows = 9, usecols = (0, 1, 2, 3, 4, 5), max_rows = 41)
ValoresT3_2 = np.loadtxt('datos_3.txt', skiprows = 56, usecols = (0, 1, 2, 3, 4), max_rows = 21)
ValoresT3_3 = np.loadtxt('datos_3.txt', skiprows = 83, usecols = (0, 1, 2, 3, 4), max_rows = 19)
ValoresT3_4 = np.loadtxt('datos_3.txt', skiprows = 107, usecols = (0, 1, 2), max_rows = 21)

##

ValoresT4_1 = np.loadtxt('Datos_EXP_2_A.txt', skiprows = 5, usecols = (0, 1), max_rows = 30)
ValoresT4_2 = np.loadtxt('Datos_EXP_2_A.txt', skiprows = 5, usecols = (2, 3), max_rows = 9)
ValoresT4_3 = np.loadtxt('Datos_EXP_2_A.txt', skiprows = 19, usecols = (2, 3), max_rows = 1)


ValoresT5_1 = np.loadtxt('Datos_EXP_2_B.txt', skiprows = 5, usecols = (0, 1), max_rows = 30)
ValoresT5_2 = np.loadtxt('Datos_EXP_2_B.txt', skiprows = 5, usecols = (2, 3), max_rows = 9)
ValoresT5_3 = np.loadtxt('Datos_EXP_2_B.txt', skiprows = 19, usecols = (2, 3), max_rows = 1)

ValoresT6_1 = np.loadtxt('Datos_EXP_2_C.txt', skiprows = 5, usecols = (0, 1), max_rows = 30)
ValoresT6_2 = np.loadtxt('Datos_EXP_2_C.txt', skiprows = 5, usecols = (2, 3), max_rows = 9)
ValoresT6_3 = np.loadtxt('Datos_EXP_2_C.txt', skiprows = 19, usecols = (2, 3), max_rows = 1)

Datos1_1 = np.zeros((6, 41))
Datos1_2 = np.zeros((5, 21))
Datos1_3 = np.zeros((5, 19))
Datos1_4 = np.zeros((3, 21))

Datos2_1 = np.zeros((6, 41))
Datos2_2 = np.zeros((5, 21))
Datos2_3 = np.zeros((5, 19))
Datos2_4 = np.zeros((3, 21))

Datos3_1 = np.zeros((6, 41))
Datos3_2 = np.zeros((5, 21))
Datos3_3 = np.zeros((5, 19))
Datos3_4 = np.zeros((3, 21))

Datos4_1 = np.zeros((2, 30))
Datos4_2 = np.zeros((2, 9))

Datos5_1 = np.zeros((2, 30))
Datos5_2 = np.zeros((2, 9))

Datos6_1 = np.zeros((2, 30))
Datos6_2 = np.zeros((2, 9))

for i in range(6):
    for j in range(41):
        Datos1_1[i][j] = ValoresT1_1[j][i]
        Datos2_1[i][j] = ValoresT2_1[j][i]
        Datos3_1[i][j] = ValoresT3_1[j][i]

for i in range(5):
    for j in range(21):
        Datos1_2[i][j] = ValoresT1_2[j][i]
        Datos2_2[i][j] = ValoresT2_2[j][i]
        Datos3_2[i][j] = ValoresT3_2[j][i]
        
for i in range(5):
    for j in range(19):
        Datos1_3[i][j] = ValoresT1_3[j][i]
        Datos2_3[i][j] = ValoresT2_3[j][i]
        Datos3_3[i][j] = ValoresT3_3[j][i]

for i in range(3):
    for j in range(21):
        Datos1_4[i][j] = ValoresT1_4[j][i]
        Datos2_4[i][j] = ValoresT2_4[j][i]
        Datos3_4[i][j] = ValoresT3_4[j][i]
        
##

for i in range(2):
    for j in range(30):
        Datos4_1[i][j] = ValoresT4_1[j][i]
        Datos5_1[i][j] = ValoresT5_1[j][i]
        Datos6_1[i][j] = ValoresT6_1[j][i]

for i in range(2):
    for j in range(9):
        Datos4_2[i][j] = ValoresT4_2[j][i]
        Datos5_2[i][j] = ValoresT5_2[j][i]
        Datos6_2[i][j] = ValoresT6_2[j][i]

valor_z = np.linspace(-0.2, 0.2, 1000)
N, I, R = 154, 2.53, 0.2

a0_A1, a1_A1, Ajustex_A1, Ajustey_A1, rA1, Ea1_A1 = MinimosCuadrados(Datos4_1[0], Datos4_1[1])
a0_B1, a1_B1, Ajustex_B1, Ajustey_B1, rB1, Ea1_B1 = MinimosCuadrados(Datos5_1[0], Datos5_1[1])
a0_C1, a1_C1, Ajustex_C1, Ajustey_C1, rC1, Ea1_C1 = MinimosCuadrados(Datos6_1[0], Datos6_1[1])

K_A = np.around(a1_A1, 2)
K_B = np.around(a1_B1, 2)
K_C = np.around(a1_C1, 2)

print('El valor K para Datos_EXP_2_A.txt es:',K_A,'+/-',Ea1_A1,'mT/A')
print('Indice de correlacion:', rA1)
print('El valor K para Datos_EXP_2_B.txt es:',K_B,'+/-',Ea1_B1,'mT/A')
print('Indice de correlacion:', rB1)
print('El valor K para Datos_EXP_2_C.txt es:',K_C,'+/-',Ea1_C1,'mT/A')
print('Indice de correlacion:', rC1)
print('')

a0_A2, a1_A2, Ajustex_A2, Ajustey_A2, rA2, Ea1_A2 = MinimosCuadrados(np.tan(Datos4_2[1]*np.pi/180), Datos4_2[0]*a1_A1)
a0_B2, a1_B2, Ajustex_B2, Ajustey_B2, rB2, Ea1_B2 = MinimosCuadrados(np.tan(Datos5_2[1]*np.pi/180), Datos5_2[0]*a1_B1)
a0_C2, a1_C2, Ajustex_C2, Ajustey_C2, rC2, Ea1_C2 = MinimosCuadrados(np.tan(Datos6_2[1]*np.pi/180), Datos6_2[0]*a1_C1)

Bhe_A, EBhe_A = a1_A2/1e-3, Ea1_A2/1e-3
Bhe_B, EBhe_B = a1_B2/1e-3, Ea1_B2/1e-3
Bhe_C, EBhe_C = a1_C2/1e-3, Ea1_C2/1e-3

print('La componente horizontal de Be para Datos_EXP_2_A.txt es',np.around(Bhe_A, 2),'+/-',np.around(EBhe_A, 2),'uT')
print('Indice de correlacion:', rA2)
print('La componente horizontal de Be para Datos_EXP_2_B.txt es',np.around(Bhe_B, 2),'+/-',np.around(EBhe_B, 2),'uT')
print('Indice de correlacion:', rB2)
print('La componente horizontal de Be para Datos_EXP_2_C.txt es',np.around(Bhe_C, 2),'+/-',np.around(EBhe_C, 2),'uT')
print('Indice de correlacion:', rC2)
print('')

a_A, Ea_A = np.mean(ValoresT4_3), np.sqrt(2*0.2**2)
a_B, Ea_B = np.mean(ValoresT5_3), np.sqrt(2*0.2**2)
a_C, Ea_C = np.mean(ValoresT6_3), np.sqrt(2*0.2**2)

print('El angulo entre componentes horizontal y vertical de Be para Datos_EXP_2_A.txt es',a_A,'+/-',np.around(Ea_A, 2))
print('El angulo entre componentes horizontal y vertical de Be para Datos_EXP_2_B.txt es',a_B,'+/-',np.around(Ea_B, 2))
print('El angulo entre componentes horizontal y vertical de Be para Datos_EXP_2_C.txt es',a_C,'+/-',np.around(Ea_C, 2))
print('')

Bve_A, EBve_A = np.tan(np.pi/180*a_A)*Bhe_A, np.sqrt((0.28/np.tan(np.pi/180*a_A))**2 + (Ea1_A2/Bhe_A*1e3)**2)
Bve_B, EBve_B = np.tan(np.pi/180*a_B)*Bhe_B, np.sqrt((0.28/np.tan(np.pi/180*a_B))**2 + (Ea1_B2/Bhe_B*1e3)**2)
Bve_C, EBve_C = np.tan(np.pi/180*a_C)*Bhe_C, np.sqrt((0.28/np.tan(np.pi/180*a_C))**2 + (Ea1_C2/Bhe_C*1e3)**2)

print('La componente vertical de Be para Datos_EXP_2_A.txt es',np.around(Bve_A, 2),'+/-',np.around(EBve_A, 2),'uT')
print('La componente vertical de Be para Datos_EXP_2_B.txt es',np.around(Bve_B, 2),'+/-',np.around(EBve_B, 2),'uT')
print('La componente vertical de Be para Datos_EXP_2_C.txt es',np.around(Bve_C, 2),'+/-',np.around(EBve_C, 2),'uT')
print('')
Be_A, EBe_A = np.sqrt(Bve_A**2 + Bhe_A**2), np.sqrt((2*EBve_A/Bve_A)**2 + (2*EBhe_A/Bhe_A)**2)
Be_B, EBe_B = np.sqrt(Bve_B**2 + Bhe_B**2), np.sqrt((2*EBve_B/Bve_B)**2 + (2*EBhe_B/Bhe_B)**2)
Be_C, EBe_C = np.sqrt(Bve_C**2 + Bhe_C**2), np.sqrt((2*EBve_C/Bve_C)**2 + (2*EBhe_C/Bhe_C)**2)

print('Be para para Datos_EXP_2_A.txt es',np.around(Be_A, 2),'+/-',np.around(EBe_A, 2),'uT')
print('Be para para Datos_EXP_2_B.txt es',np.around(Be_B, 2),'+/-',np.around(EBe_B, 2),'uT')
print('Be para para Datos_EXP_2_C.txt es',np.around(Be_C, 2),'+/-',np.around(EBe_C, 2),'uT')

plt.figure(figsize = (10, 8))
plt.plot(Datos1_1[0]/100, Datos1_1[1], 'k--')
plt.plot(Datos1_1[2]/100, Datos1_1[3], 'k-.')
plt.plot(Datos1_1[4]/100, Datos1_1[5], 'k-')
plt.legend([r'$\alpha = R$', r'$\alpha = R/2$', r'$\alpha = 2R$'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos1_2[0]/100, Datos1_2[1], 'k--')
plt.plot(Datos1_2[0]/100, Datos1_2[2], 'k-.')
plt.plot(Datos1_2[0]/100, Datos1_2[3], 'k-d')
plt.plot(Datos1_2[0]/100, Datos1_2[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_z$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos1_3[0]/100, Datos1_3[1], 'k--')
plt.plot(Datos1_3[0]/100, Datos1_3[2], 'k-.')
plt.plot(Datos1_3[0]/100, Datos1_3[3], 'k-d')
plt.plot(Datos1_3[0]/100, Datos1_3[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos1_4[0]/100, Datos1_4[1], 'k--')
plt.plot(Datos1_4[0]/100, Datos1_4[2], 'k-.')
plt.legend(['Bobina 1', 'Bobina 2'], fontsize = 15)
plt.xlabel(r'$r$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

##

plt.figure(figsize = (10, 8))
plt.plot(Datos2_1[0]/100, Datos2_1[1], 'k--')
plt.plot(Datos2_1[2]/100, Datos2_1[3], 'k-.')
plt.plot(Datos2_1[4]/100, Datos2_1[5], 'k-')
plt.legend([r'$\alpha = R$', r'$\alpha = R/2$', r'$\alpha = 2R$'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos2_2[0]/100, Datos2_2[1], 'k--')
plt.plot(Datos2_2[0]/100, Datos2_2[2], 'k-.')
plt.plot(Datos2_2[0]/100, Datos2_2[3], 'k-d')
plt.plot(Datos2_2[0]/100, Datos2_2[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_z$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos2_3[0]/100, Datos2_3[1], 'k--')
plt.plot(Datos2_3[0]/100, Datos2_3[2], 'k-.')
plt.plot(Datos2_3[0]/100, Datos2_3[3], 'k-d')
plt.plot(Datos2_3[0]/100, Datos2_3[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos2_4[0]/100, Datos2_4[1], 'k--')
plt.plot(Datos2_4[0]/100, Datos2_4[2], 'k-.')
plt.legend(['Bobina 1', 'Bobina 2'], fontsize = 15)
plt.xlabel(r'$r$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

###

plt.figure(figsize = (10, 8))
plt.plot(Datos3_1[0]/100, Datos3_1[1], 'k--')
plt.plot(Datos3_1[2]/100, Datos3_1[3], 'k-.')
plt.plot(Datos3_1[4]/100, Datos3_1[5], 'k-')
plt.legend([r'$\alpha = R$', r'$\alpha = R/2$', r'$\alpha = 2R$'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos3_2[0]/100, Datos3_2[1], 'k--')
plt.plot(Datos3_2[0]/100, Datos3_2[2], 'k-.')
plt.plot(Datos3_2[0]/100, Datos3_2[3], 'k-d')
plt.plot(Datos3_2[0]/100, Datos3_2[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_z$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos3_3[0]/100, Datos3_3[1], 'k--')
plt.plot(Datos3_3[0]/100, Datos3_3[2], 'k-.')
plt.plot(Datos3_3[0]/100, Datos3_3[3], 'k-d')
plt.plot(Datos3_3[0]/100, Datos3_3[4], 'k-')
plt.legend([r'$r = 0$ m', r'$r = 0.1$ m', r'$r = 0.14$ m', r'$r = 0.16$ m'], fontsize = 15)
plt.xlabel(r'$z$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Datos3_4[0]/100, Datos3_4[1], 'k--')
plt.plot(Datos3_4[0]/100, Datos3_4[2], 'k-.')
plt.legend(['Bobina 1', 'Bobina 2'], fontsize = 15)
plt.xlabel(r'$r$ (m)', fontsize = 15)
plt.ylabel(r'$B_r$ (mT)', fontsize = 15)
plt.grid()

##

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_A1, Ajustey_A1, 'k--')
plt.plot(Datos4_1[0], Datos4_1[1], 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$I_H$ (A)', fontsize = 15)
plt.ylabel(r'$B_{H_h}$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_A2, Ajustey_A2, 'k--')
plt.plot(np.tan(Datos4_2[1]*np.pi/180), Datos4_2[0]*a1_A1, 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$\tan(\alpha)$', fontsize = 15)
plt.ylabel(r'$I_H\cdot K$ (mT)', fontsize = 15)
plt.grid()

##

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_B1, Ajustey_B1, 'k--')
plt.plot(Datos5_1[0], Datos5_1[1], 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$I_H$ (A)', fontsize = 15)
plt.ylabel(r'$B_{H_h}$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_B2, Ajustey_B2, 'k--')
plt.plot(np.tan(Datos5_2[1]*np.pi/180), Datos5_2[0]*a1_B1, 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$\tan(\alpha)$', fontsize = 15)
plt.ylabel(r'$I_H\cdot K$ (mT)', fontsize = 15)
plt.grid()

##

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_C1, Ajustey_C1, 'k--')
plt.plot(Datos6_1[0], Datos6_1[1], 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$I_H$ (A)', fontsize = 15)
plt.ylabel(r'$B_{H_h}$ (mT)', fontsize = 15)
plt.grid()

plt.figure(figsize = (10, 8))
plt.plot(Ajustex_C2, Ajustey_C2, 'k--')
plt.plot(np.tan(Datos6_2[1]*np.pi/180), Datos6_2[0]*a1_C1, 'ko')
plt.legend(['Ajuste', 'Puntos'], fontsize = 15)
plt.xlabel(r'$\tan(\alpha)$', fontsize = 15)
plt.ylabel(r'$I_H\cdot K$ (mT)', fontsize = 15)
plt.grid()

plt.show()