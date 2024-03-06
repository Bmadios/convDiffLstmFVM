import numpy as np
import torch

def lossFVMDiscretizedEqns(T, Tp, Nx, Ny, dx, dy, dS, dt, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs):
    #T = np.ones((Ny, Nx))
    total_loss = 0  # Initialize total loss
    for i in range(Ny):
        for j in range(Nx):
            aP0 = dS/dt
            dF = ( (Fe - Fw) + (Fn - Fs) )
            iTop, iBot = -1, 0
            jRight, jLeft = -1, 0 
                
            if (j == 0 and i > 0 and i < Ny - 1): # LEFT Wall (1ere colonne)
                aE = De/dx - Fe/2
                aW = 0
                aN = Dn/dy - Fn/2
                aS = Ds/dy + Fs/2
                    
                sP = - (Dw/(dx/2) + Fw)
                sU = (Dw/(dx/2) + Fw)*T[i, jLeft]
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, jLeft]
                V = aN*T[i+1, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])
                    
            elif (j == Nx - 1 and i > 0 and i < Ny - 1): # RIGHT WALL (Nx - 1 colonne)
                aE = 0
                aW = Dw/dx + Fw/2
                aN = Dn/dy - Fn/2
                aS = Ds/dy + Fs/2
                    
                sP = - (De/(dx/2) - Fe)
                sU = (De/(dx/2) - Fe)*T[i, jRight]
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, jRight] + aW*T[i, j-1]
                V = aN*T[i+1, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])
                    
                    
            elif (i == 0 and j > 0 and j < Nx - 1): # BOTTOM (1ere ligne de la matrice)
                aE = De/dx - Fe/2
                aW = Dw/dx + Fw/2
                aN = Dn/dy - Fn/2
                aS = 0
                    
                sP = - (Ds/(dy/2) + Fs)
                sU = (Ds/(dy/2) + Fs)*T[iBot, j]
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, j-1]
                V = aN*T[i+1, j] + aS*T[iBot, j]
                b = sU + (aP0*Tp[i, j])
                    
            elif (i == Ny - 1 and j > 0 and j < Nx - 1): # TOP (Ny - 1 ligne de la matrice)
                aE = De/dx - Fe/2
                aW = Dw/dx + Fw/2
                aN = 0
                aS = Ds/dy + Fs/2
                    
                sP = - (Dn/(dy/2) - Fn)
                sU = (Dn/(dy/2) - Fn)*T[iTop, j]
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, j-1]
                V = aN*T[iTop, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])
                    
            elif (i==0 and j==0): # LEFT BOTTOM CORNER
                aE = De/dx - Fe/2
                aW = 0
                aN = Dn/dy - Fn/2
                aS = 0
                    
                sP = -(Dw/(dx/2) + Ds/(dy/2) + Fw + Fs)
                sU = ((Dw/(dx/2) + Fw)*T[i, jLeft] + (Ds/(dy/2) + Fs)*T[iBot, j])
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, jLeft]
                V = aN*T[i+1, j] + aS*T[iBot, j]
                b = sU + (aP0*Tp[i, j])

                    
            elif (i==Ny-1 and j==0): # LEFT TOP CORNER
                aE = De/dx - Fe/2
                aW = 0
                aN = 0
                aS = Ds/dy + Fs/2
                    
                sP = -(Dw/(dx/2) + Dn/(dy/2) + Fw - Fn)
                sU = ((Dw/(dx/2) + Fw)*T[i, jLeft] + (Dn/(dy/2) - Fn)*T[iTop, j])
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, jLeft]
                V = aN*T[iTop, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])
 
                
            elif (i==0 and j==Nx-1): # RIGHT BOTTOM CORNER
                aE = 0
                aW = Dw/dx + Fw/2
                aN = Dn/dy - Fn/2
                aS = 0
                    
                sP = -(De/(dx/2) + Ds/(dy/2) - Fe + Fs)
                sU = ((De/(dx/2) - Fe)*T[i, jRight] + (Ds/(dy/2) + Fs)*T[iBot, j])
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, jRight] + aW*T[i, j-1]
                V = aN*T[i+1, j] + aS*T[iBot, j]
                b = sU + (aP0*Tp[i, j])

                
            elif (i==Ny-1 and j==Ny-1): # RIGHT top CORNER
                aE = 0
                aW = Dw/dx + Fw/2
                aN = 0
                aS = Ds/dy + Fs/2
                    
                sP = -(De/(dx/2) + Dn/(dy/2) - Fe - Fn)
                sU = ((De/(dx/2) - Fe)*T[i, jRight] + (Dn/(dy/2) - Fn)*T[iTop, j])
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, jRight] + aW*T[i, j-1]
                V = aN*T[iTop, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])
                    
            else: # INTERNAL CELLS
                aE = De/dx - Fe/2
                aW = Dw/dx + Fw/2
                aN = Dn/dy - Fn/2
                aS = Ds/dy + Fs/2
                    
                sP = 0
                sU = 0
                aP = aP0 + aE + aW + aN + aS - sP + dF
                    
                H = aE*T[i, j+1] + aW*T[i, j-1]
                V = aN*T[i+1, j] + aS*T[i-1, j]
                b = sU + (aP0*Tp[i, j])

            #print(aP)
            #print(H)
            #print(V)
            #T[i,j] = 1/aP*(H + V + b)
            residuals = T[i, j]*aP - (H+V+b)
            total_loss += residuals**2  # Sum of squared residuals
        
        return torch.mean(total_loss)
        #return T