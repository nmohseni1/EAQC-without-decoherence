import time
import numpy as np
from qutip import *
from math import sqrt
from scipy import *
import matplotlib.pyplot as plt
##*******************************      set up the parapeteres
count_time=0
mat1 = genfromtxt("test");
for taumax in range (100,101,20):
    count_time=count_time+1
    print('taulist',taumax)
    taulist= np.linspace(0,taumax,100)
    num_trial=850                     #number of instances
    p1_final=[]
    random.seed(10)
    q=8                                 #number of particles
    for N in range(1,q):
        M=(N+1)*(N+1)*(N+1)
        probability=[]
        print('N=',N)
        a=destroy(N+1)
    #********************************** Construct the  Initial State
        g1=np.zeros((N+1,1),float)
        for i in range(0,N+1):
            g1[i]=((1./sqrt(2))**i)*((1./sqrt(2))**(N-i))*sqrt(math.factorial(N)/((math.factorial(i))*(math.factorial(N-i))))
        psi_list=[Qobj(g1).unit() for n in range(3)]
        psii=tensor(psi_list)
    #*********************************  Construct the Hamiltonian
        I1=qeye(N+1)
        z=(2*( a.dag()*a)-(N)*I1)
        x=np.zeros((N+1,N+1), complex)
        for i in range (0,N):
            x[i+1][i]=np.sqrt(N-i)*np.sqrt(i+1)
            x[i][i+1]=np.sqrt(N-i)*np.sqrt(i+1)
        y=np.zeros((N+1,N+1),complex)
        for i in range (0,N):
            y[i+1][i]=-1j*np.sqrt(N-i)*np.sqrt(i+1)
            y[i][i+1]=1j*np.sqrt(N-i)*np.sqrt(i+1)
        I=tensor(qeye(N+1),qeye(N+1),qeye(N+1))
        sz1=tensor(Qobj(z),qeye(N+1),qeye(N+1))
        sz2=tensor(qeye(N+1),Qobj(z),qeye(N+1))
        sz3=tensor(qeye(N+1),qeye(N+1),Qobj(z))
        sx1=tensor(Qobj(x),qeye(N+1),qeye(N+1))
        sx2=tensor(qeye(N+1),Qobj(x),qeye(N+1))
        sx3=tensor(qeye(N+1),qeye(N+1),Qobj(x))
        H0=-(sx1+sx2+sx3)
        count=0
        random.seed(10)
      #*********************************************************
        success_eachtrial=[]
        regime2=[]
        for i in range(1,num_trial+1):
            count=count+1
            #print('count',count)
            number=[]
            number1=[]
            def H1(N):
                Jmat=np.random.uniform(low=-1, high=1, size=(3,3) )
                Jmat=Jmat+Jmat.transpose()
                for i in range(0,len(Jmat)):
                    Jmat[i,i]=0
                #print (Jmat)
                Kmat=np.random.uniform(low=-1, high=1, size=(1,3) )
                #print(Kmat)

                #Jmat[1,0]=1#-0.50164328         #for the case that we consider just one instance
                #Jmat[0,1]=1#-0.50164328
                #Jmat[2,0]=0#0212#-0.30679039
                #Jmat[0,2]=0#0.0212#-0.30679039
                #Jmat[1,2]=1#0.853#-0.97973137
                #Jmat[2,1]=1#0.853#-0.97973137

                #Kmat[0,0]=0#-0.8876
                #Kmat[0,1]=2#0.7686
                #Kmat[0,2]=1#0.9711
                H1=sum(Jmat[i-1,j-1]*eval("sz"+str(i))*eval("sz"+str(j))*(1./N) for i in range(1,len(Jmat)+1) for j in range(1,4))+sum(Kmat[0,i-1]*eval("sz"
                   +str(i)) for i in range(1,4))
                return(H1,Jmat,Kmat)
            H11,J1,K1=H1(N)
            evals1, ekets1=H11.eigenstates()  #Just to check if the final eigensystem is the same with output of Adiabtic evolution
            #print('H11',H11)
            #print('ekets1',ekets1)
            #print('evals1',evals1)
            exx_1=np.zeros((1,M))
            exx_2=np.zeros((1,M))
            exx_3=np.zeros((1,M))
            for n, eket in enumerate(ekets1):  # Required for Calculating the number of Replica states
                exx_1[0,n]=expect(sz1,ekets1[n])
                exx_2[0,n]=expect(sz2,ekets1[n])
                exx_3[0,n]=expect(sz3,ekets1[n])
            args = {'t_max': max(taulist)}
            h_t = [[H0, lambda t, args :( 1-(t/args['t_max']))],
                     [H11, lambda t, args : (t/args['t_max'])]]
            if count in mat1:                 #call the set in each regimes
                #print('count_new',count)
    #***************************************** set parameters
                evals_mat = np.zeros((len(taulist),M))
                P_n=np.zeros((len(taulist),M))
                P_mat = np.zeros((len(taulist),M))
                ex_1=np.zeros((len(taulist),M))
                ex_2=np.zeros((len(taulist),M))
                ex_3=np.zeros((len(taulist),M))
                exx_x=np.zeros((len(taulist),M))
                success_probability=np.zeros((len(taulist),M))
                pho=[]
                idx = [0]
                def process_rho(tau, psi):

    #**************************************** evaluate the Hamiltonian with gradually switched on interaction
                    H = qobj_list_evaluate(h_t, tau, args)

    #***************************************   eigenvalues of the Hamiltonian
                    evals, ekets = H.eigenstates(eigvals=M)
                    evals_mat[idx[0],:] = real(evals)
    #***************************************  occupation probabilities of the energy levels without decoherence

                    for n, eket in enumerate(ekets):
                        P_mat[idx[0],n] =abs((eket.dag().data * psi.data)[0,0])**2
    #***********************************  occupation probability of the energy levels with decoherence
                        #P_mat[idx[0],n] =(abs((ekets[n].dag() * psi*ekets[n])[0,0]))
                      #x=x+P_mat[idx[0],n]  #just to check if sum(P)=1
    #**************************************** expectation values of sz operators (for finding replica states)
                        ex_1[idx[0],n]=expect(sz1,ekets[n])
                        ex_2[idx[0],n]=expect(sz2,ekets[n])
                        ex_3[idx[0],n]=expect(sz3,ekets[n])
     #*************************************** calculating |<psi|n>|^2 |n>:eigenstates of sz
                    count_1=0
                    for  i in range(N+1):
                        for j in range(N+1):
                            for k in range(N+1):
                                count_1=count_1+1
                                P_n[idx[0],count_1-1]=abs((psi.dag()*tensor(fock(N+1,i),fock(N+1,j),fock(N+1,k)))[0,0])**2
                    idx[0] += 1
                    return(psi)
    #**************************************** solving the master equation without and with decoherence(Forth argument is related to deohernce operators)

                mesolve(h_t, psii, taulist, [], process_rho, args,_safe_mode=False)
    #***********************************  finding the number of equiavalent states with ground state

                for n in range(M):        #using the output of the adiabatic evolution
                    if ex_1[len(taulist)-1,0]*ex_1[len(taulist)-1,n]<=0 or  ex_2[len(taulist)-1,0]*ex_2[len(taulist)-1,n]<=0 or ex_3[len(taulist)-1,0]*ex_3[len(taulist)-1,n]<=0:
                        number.append(n)
                for g in range(M):         #directly from H1
                    if exx_1[0,0]*exx_1[0,g]<=0 or  exx_2[0,0]*exx_2[0,g]<=0 or exx_3[0,0]*exx_3[0,g]<=0:
                        number1.append(g)
                ng=number[0]                #number of replica states
                print('ng',number[0])
                #print('nf',number1[0])
                success_probability=(sum(P_mat[len(taulist)-1,n] for n in range(0,ng))) #put ng=0 for the first method
                success_eachtrial.append(success_probability)#log_success=log(10*(1-success_probability))

        #print('success_eachtrial',success_eachtrial)
        #print('success_eachtrial',size(success_eachtrial))
        success_alltrials=sum(success_eachtrial[i] for i in range(60))
        #print('success_alltrials',success_alltrials)

        p1_final.append(success_alltrials/60)

        print('p_final',p1_final)

     #********************************   save success probability

        P_success='P_instancesregime1'+str(count_time)
        P_success_f=open(P_success,'w')
        for i in range(1):
            tt=str(i+1)+"\t"+str(p1_final[i])+"\n"
            P_success_f.write(tt)
