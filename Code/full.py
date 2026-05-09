import numpy as np
import random
import math
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ============================================================
# SEED CỐ ĐỊNH  →  mỗi lần chạy kết quả LUÔN GIỐNG NHAU
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# DỮ LIỆU SOLOMON C101
# ============================================================
depot     = (40, 50)
depot_due = 1236
Q         = 200

customers = [
    (45,68,10,912,967,90),(45,70,30,825,870,90),(42,66,10,65,146,90),
    (42,68,10,727,782,90),(42,65,10,15,67,90),(40,69,20,621,702,90),
    (40,66,20,170,225,90),(38,68,20,255,324,90),(38,70,10,534,605,90),
    (35,66,10,357,410,90),(35,69,10,448,505,90),(25,85,20,652,721,90),
    (22,75,30,30,92,90),(22,85,10,567,620,90),(20,80,40,384,429,90),
    (20,85,40,475,528,90),(18,75,20,99,148,90),(15,75,20,179,254,90),
    (15,80,10,278,345,90),(30,50,10,10,73,90),(30,52,20,914,965,90),
    (28,52,20,812,883,90),(28,55,10,732,777,90),(25,50,10,65,144,90),
    (25,52,40,169,224,90),(25,55,10,622,701,90),(23,52,10,261,316,90),
    (23,55,20,546,593,90),(20,50,10,358,405,90),(20,55,10,449,504,90),
    (10,35,20,200,237,90),(10,40,30,31,100,90),(8,40,40,87,158,90),
    (8,45,20,751,816,90),(5,35,10,283,344,90),(5,45,10,665,716,90),
    (2,40,20,383,434,90),(0,40,30,479,522,90),(0,45,20,567,624,90),
    (35,30,10,264,321,90),(35,32,10,166,235,90),(33,32,20,68,149,90),
    (33,35,10,16,80,90),(32,30,10,359,412,90),(30,30,10,541,600,90),
    (30,32,30,448,509,90),(30,35,10,1054,1127,90),(28,30,10,632,693,90),
    (28,35,10,1001,1066,90),(26,32,10,815,880,90),(25,30,10,725,786,90),
    (25,35,10,912,969,90),(44,5,20,286,347,90),(42,10,40,186,257,90),
    (42,15,10,95,158,90),(40,5,30,385,436,90),(40,15,40,35,87,90),
    (38,5,30,471,534,90),(38,15,10,651,740,90),(35,5,20,562,629,90),
    (50,30,10,531,610,90),(50,35,20,262,317,90),(50,40,50,171,218,90),
    (48,30,10,632,693,90),(48,40,10,76,129,90),(47,35,10,826,875,90),
    (47,40,10,12,77,90),(45,30,10,734,777,90),(45,35,10,916,969,90),
    (95,30,30,387,456,90),(95,35,20,293,360,90),(53,30,10,450,505,90),
    (92,30,10,478,551,90),(53,35,50,353,412,90),(45,65,20,997,1068,90),
    (90,35,10,203,260,90),(88,30,10,574,643,90),(88,35,20,109,170,90),
    (87,30,10,668,731,90),(85,25,10,769,820,90),(85,35,30,47,124,90),
    (75,55,20,369,420,90),(72,55,10,265,338,90),(70,58,20,458,523,90),
    (68,60,30,555,612,90),(66,55,10,173,238,90),(65,55,20,85,144,90),
    (65,60,30,645,708,90),(63,58,10,737,802,90),(60,55,10,20,84,90),
    (60,60,10,836,889,90),(67,85,20,368,441,90),(65,85,40,475,518,90),
    (65,82,10,285,336,90),(62,80,30,196,239,90),(60,80,10,95,156,90),
    (60,85,30,561,622,90),(58,75,20,30,84,90),(55,80,10,743,820,90),
    (55,85,20,647,726,90),
]
N = len(customers)

# ============================================================
# DISTANCE MATRIX  (node 0=depot, node i+1=customer i)
# ============================================================
def _build_D():
    pts = [depot]+[(c[0],c[1]) for c in customers]
    n   = len(pts); D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dx=pts[i][0]-pts[j][0]; dy=pts[i][1]-pts[j][1]
            D[i][j]=math.sqrt(dx*dx+dy*dy)
    return D
D = _build_D()

# ============================================================
# CORE DECODE / FITNESS
# ============================================================
def perm_to_sol(perm):
    """Decode permutation → list of routes (VRPTW constraints)."""
    routes=[]; route=[]; load=0; cn=0; ct=0.0
    for idx in perm:
        _,_,dem,rdy,due,svc = customers[idx]
        need=load+dem>Q
        if not need:
            tr=D[cn][idx+1]; arr=max(ct+tr,float(rdy))
            if arr>due: need=True
        if need:
            if route: routes.append(route)
            route=[]; load=0; cn=0; ct=0.0
        tr=D[cn][idx+1]; arr=max(ct+tr,float(rdy))
        route.append(idx); load+=dem; ct=arr+svc; cn=idx+1
    if route: routes.append(route)
    return routes

def sol_to_perm(sol):
    p=[]
    for r in sol: p.extend(r)
    return p

def route_dist(route):
    if not route: return 0.0
    d=D[0][route[0]+1]
    for i in range(len(route)-1): d+=D[route[i]+1][route[i+1]+1]
    d+=D[route[-1]+1][0]
    return d

def sol_fitness(sol):  return sum(route_dist(r) for r in sol)
def fitness(perm):     return sol_fitness(perm_to_sol(perm))
def count_vehicles(p): return len(perm_to_sol(p))

# ============================================================
# NEAREST NEIGHBOR (VRPTW-aware)
# ============================================================
def nn_sol(noise=0.0):
    unvisited=list(range(N)); routes=[]
    while unvisited:
        route=[]; load=0; cn=0; ct=0.0; changed=True
        while changed and unvisited:
            changed=False; cands=[]
            for c in unvisited:
                _,_,d,rdy,due,svc=customers[c]
                if load+d>Q: continue
                tr=D[cn][c+1]; arr=max(ct+tr,float(rdy))
                if arr>due: continue
                if arr+svc+D[c+1][0]>depot_due: continue
                cands.append((tr,c))
            if cands:
                cands.sort()
                top=cands[:max(1,int(len(cands)*0.3)+1)]
                _,bc=random.choice(top) if noise>0 and random.random()<noise and len(top)>1 else top[0]
                _,_,d,rdy,due,svc=customers[bc]
                arr=max(ct+D[cn][bc+1],float(rdy))
                route.append(bc); unvisited.remove(bc)
                load+=d; ct=arr+svc; cn=bc+1; changed=True
        routes.append(route)
    return routes

# ============================================================
# LOCAL SEARCH
# ============================================================
def two_opt(route):
    if len(route)<3: return route
    best=route[:]; bd=route_dist(best); imp=True
    while imp:
        imp=False
        for i in range(len(best)-1):
            for j in range(i+2,len(best)):
                nr=best[:i]+best[i:j+1][::-1]+best[j+1:]
                d=route_dist(nr)
                if d<bd-1e-6: best,bd,imp=nr,d,True
    return best

def route_feasible(route):
    load=0; cn=0; ct=0.0
    for c in route:
        _,_,d,rdy,due,svc=customers[c]
        if load+d>Q: return False
        arr=max(ct+D[cn][c+1],float(rdy))
        if arr>due: return False
        if arr+svc+D[c+1][0]>depot_due: return False
        load+=d; ct=arr+svc; cn=c+1
    return True

def or_opt(sol):
    """Or-opt: di chuyển 1 khách hàng sang route khác nếu giảm cost."""
    sol=[r[:] for r in sol]; imp=True
    while imp:
        imp=False
        for i in range(len(sol)):
            for pos in range(len(sol[i])):
                c=sol[i][pos]
                new_ri=sol[i][:pos]+sol[i][pos+1:]
                best_gain=1e-6; bj=-1; bpos=0
                for j in range(len(sol)):
                    for ins in range(len(sol[j])+(0 if j==i else 1)):
                        if j==i and ins==pos: continue
                        nr=sol[j][:ins]+[c]+sol[j][ins:]
                        if not route_feasible(nr): continue
                        gain=(route_dist(sol[i])+route_dist(sol[j])
                              -route_dist(new_ri)-route_dist(nr))
                        if gain>best_gain:
                            best_gain=gain; bj=j; bpos=ins
                if bj>=0:
                    sol[i]=new_ri; sol[bj]=sol[bj][:bpos]+[c]+sol[bj][bpos:]
                    sol=[r for r in sol if r]; imp=True; break
            if imp: break
    return sol

def local_search(sol, do_or=True):
    sol=[two_opt(r) for r in sol if r]
    if do_or: sol=or_opt(sol)
    return [r for r in sol if r]

# ============================================================
# GENETIC ALGORITHM
# OX crossover + RSM mutation trên permutation (nhanh, O(n))
# Periodic or-opt trên top-k để cải thiện chất lượng
# Tham số bài báo: pop=100, mr=0.5, ts=0.2
# ============================================================
def ox_crossover(p1, p2):
    """Order Crossover (OX) - O(n), giữ thứ tự tương đối."""
    s = len(p1)
    a, b = sorted(random.sample(range(s), 2))
    child = [-1]*s
    child[a:b] = p1[a:b]
    in_child = set(child[a:b])
    fill = [x for x in p2 if x not in in_child]
    fi = 0
    for i in list(range(b, s)) + list(range(0, a)):
        child[i] = fill[fi]; fi += 1
    return child

def rsm(ind):
    """Reverse Sequence Mutation."""
    ind = list(ind); i,j = sorted(random.sample(range(len(ind)),2))
    ind[i:j+1] = ind[i:j+1][::-1]; return ind

def run_GA(pop_size=100, generations=500, mr=0.5, ts=0.2, seed=42):
    random.seed(seed); np.random.seed(seed)
    print("\n"+"="*58)
    print(f"  GENETIC ALGORITHM  [pop={pop_size}, gen={generations}]")
    print("="*58)
    t0 = time.time()

    # Init: NN heuristic → permutation (bài báo)
    pop = [sol_to_perm(nn_sol(0.0))]
    for _ in range(pop_size-1):
        pop.append(sol_to_perm(nn_sol(0.4)))

    fits = [fitness(p) for p in pop]
    bi = int(np.argmin(fits)); best = pop[bi][:]; bf = fits[bi]
    history = [bf]
    print(f"  Init best: {bf:.2f}")

    for gen in range(generations):
        # Truncation Selection + FPS (bài báo ts=0.2)
        order = sorted(range(pop_size), key=lambda i: fits[i])
        k = max(2, int(pop_size*ts))
        cands = [pop[i] for i in order[:k]]
        cfs   = [fits[i] for i in order[:k]]
        mxf   = max(cfs)
        w = np.array([mxf-f+1e-6 for f in cfs]); w /= w.sum()

        # Elitism top-2
        new_pop  = [pop[order[0]][:], pop[order[1]][:]]
        new_fits = [fits[order[0]],   fits[order[1]]]

        for _ in range(pop_size-2):
            i1 = int(np.random.choice(k, p=w))
            i2 = int(np.random.choice(k, p=w))
            child = ox_crossover(cands[i1], cands[i2])
            if random.random() < mr:
                child = rsm(child)
            new_pop.append(child)
            new_fits.append(fitness(child))

        pop = new_pop; fits = new_fits
        gb  = min(fits)
        if gb < bf: bf = gb; best = pop[fits.index(gb)][:]
        history.append(bf)

        # Periodic or-opt on current best (every 50 gen)
        if (gen+1) % 50 == 0:
            imp = or_opt(perm_to_sol(best))
            id2 = sol_fitness(imp)
            if id2 < bf: bf = id2; best = sol_to_perm(imp)
            history[-1] = bf

        if (gen+1) % 100 == 0:
            print(f"  Gen {gen+1:4d}/{generations} | Best={bf:.2f} | Veh={count_vehicles(best)}")

    # Final or-opt polish
    final = or_opt(perm_to_sol(best)); fd = sol_fitness(final)
    if fd < bf: bf = fd; best = sol_to_perm(final)
    history.append(bf)

    elapsed = time.time()-t0
    print(f"\n[GA] Best: {bf:.2f} | Vehicles: {count_vehicles(best)} | Time: {elapsed:.1f}s")
    return best, history, elapsed

# ============================================================
# ANT COLONY SYSTEM
# Tham số bài báo: colony=100, rho=0.3, q0=0.1, beta=2, top-k=1
# ============================================================
def run_ACS(n_ants=100, n_iter=500, rho=0.3, q0=0.1, beta=2, seed=42):
    random.seed(seed); np.random.seed(seed)
    print("\n"+"="*58)
    print(f"  ANT COLONY SYSTEM  [ants={n_ants}, iter={n_iter}]")
    print("="*58)
    t0=time.time()

    init_sol=nn_sol(0.0); L_nn=sol_fitness(init_sol)
    tau_0=1.0/(N*L_nn); nn=N+1
    pher=np.full((nn,nn),tau_0)
    eta=np.zeros((nn,nn))
    for i in range(nn):
        for j in range(nn):
            if i!=j: eta[i][j]=1.0/(D[i][j]+1e-10)

    best_sol=[r[:] for r in init_sol]; best_d=L_nn
    history=[best_d]
    print(f"  NN init: {L_nn:.2f}, tau_0={tau_0:.2e}")

    for it in range(n_iter):
        for ant in range(n_ants):
            routes=[]; route=[]; load=0; cn=0; ct=0.0
            unvisited=list(range(N))
            while unvisited:
                feasible=[]
                for c in unvisited:
                    _,_,d,rdy,due,svc=customers[c]
                    if load+d>Q: continue
                    tr=D[cn][c+1]; arr=ct+tr
                    if arr<rdy: arr=rdy
                    if arr>due: continue
                    if arr+svc+D[c+1][0]>depot_due: continue
                    feasible.append(c)
                if not feasible:
                    if route: routes.append(route)
                    route=[]; load=0; cn=0; ct=0.0
                    feasible=[c for c in unvisited if customers[c][2]<=Q]
                    if not feasible: feasible=unvisited[:1]
                q=random.random()
                nodes=[c+1 for c in feasible]
                sc=pher[cn,nodes]*(eta[cn,nodes]**beta)
                if q<=q0:
                    nx=nodes[int(np.argmax(sc))]
                else:
                    tot=sc.sum()
                    if tot==0: nx=random.choice(nodes)
                    else: nx=nodes[int(np.random.choice(len(nodes),p=sc/tot))]
                nc=nx-1
                pher[cn,nx]=(1-rho)*pher[cn,nx]+rho*tau_0
                pher[nx,cn]=pher[cn,nx]
                _,_,d,rdy,due,svc=customers[nc]
                arr=max(ct+D[cn][nx],float(rdy))
                route.append(nc); unvisited.remove(nc)
                load+=d; ct=arr+svc; cn=nx
            if route: routes.append(route)
            dd=sol_fitness(routes)
            if dd<best_d: best_d=dd; best_sol=[r[:] for r in routes]

        # Global update (Eq.15)
        pher*=(1-rho); delta=rho/best_d
        for route in best_sol:
            if not route: continue
            pher[0,route[0]+1]+=delta; pher[route[0]+1,0]+=delta
            for i in range(len(route)-1):
                pher[route[i]+1,route[i+1]+1]+=delta
                pher[route[i+1]+1,route[i]+1]+=delta
            pher[route[-1]+1,0]+=delta; pher[0,route[-1]+1]+=delta
        pher=np.maximum(pher,tau_0*0.01)

        # Periodic or-opt (mỗi 50 iter)
        if (it+1)%50==0:
            imp=or_opt(best_sol)
            id2=sol_fitness(imp)
            if id2<best_d: best_d=id2; best_sol=imp

        history.append(best_d)
        if (it+1)%100==0:
            print(f"  Iter {it+1:4d}/{n_iter} | Best={best_d:.2f} | Veh={len(best_sol)}")

    # Final polish
    final=or_opt(best_sol); fd2=sol_fitness(final)
    if fd2<best_d: best_d=fd2; best_sol=final

    elapsed=time.time()-t0
    print(f"\n[ACS] Best: {best_d:.2f} | Vehicles: {len(best_sol)} | Time: {elapsed:.1f}s")
    return sol_to_perm(best_sol), history, elapsed

# ============================================================
# PARTICLE SWARM OPTIMIZATION
# Tham số bài báo: swarm=100, w=0.1 (permutation-based)
# ============================================================
def pso_vel(a_perm, b_perm):
    a=list(a_perm); swaps=[]
    for i in range(len(a)):
        if a[i]!=b_perm[i]:
            j=a.index(b_perm[i]); swaps.append((i,j)); a[i],a[j]=a[j],a[i]
    return swaps

def pso_move(perm, swaps, prob):
    p=list(perm)
    for i,j in swaps:
        if random.random()<prob: p[i],p[j]=p[j],p[i]
    return p

def run_PSO(pop_size=100, max_iter=500, w=0.1, seed=42):
    random.seed(seed); np.random.seed(seed)
    print("\n"+"="*58)
    print(f"  PARTICLE SWARM OPTIMIZATION  [pop={pop_size}, iter={max_iter}]")
    print("="*58)
    t0=time.time()

    pos=[sol_to_perm(nn_sol(0.0))]
    for _ in range(pop_size-1):
        pos.append(sol_to_perm(nn_sol(0.5)))
    vel=[[] for _ in range(pop_size)]
    pb=[p[:] for p in pos]
    pf=[fitness(p) for p in pb]
    gi=int(np.argmin(pf))
    gb=pb[gi][:]; gf=pf[gi]
    history=[gf]
    print(f"  Init best: {gf:.2f}")

    for it in range(max_iter):
        for i in range(pop_size):
            r1=random.random(); r2=random.random()
            ss_p=pso_vel(pos[i],pb[i])
            ss_g=pso_vel(pos[i],gb)
            p=list(pos[i])
            for ii,jj in vel[i]:
                if random.random()<w: p[ii],p[jj]=p[jj],p[ii]
            for ii,jj in ss_p:
                if random.random()<r1: p[ii],p[jj]=p[jj],p[ii]
            for ii,jj in ss_g:
                if random.random()<r2: p[ii],p[jj]=p[jj],p[ii]
            vel[i]=(ss_p+ss_g)[:12]; pos[i]=p
            f=fitness(p)
            if f<pf[i]: pf[i]=f; pb[i]=p[:]
            if f<gf: gf=f; gb=p[:]

        history.append(gf)
        if (it+1)%100==0:
            print(f"  Iter {it+1:4d}/{max_iter} | Best={gf:.2f} | Veh={count_vehicles(gb)}")

    # Post or-opt
    final=or_opt(perm_to_sol(gb)); fd=sol_fitness(final)
    if fd<gf: gf=fd; gb=sol_to_perm(final)
    history.append(gf)

    elapsed=time.time()-t0
    print(f"\n[PSO] Best: {gf:.2f} | Vehicles: {count_vehicles(gb)} | Time: {elapsed:.1f}s")
    return gb, history, elapsed

# ============================================================
# VISUALIZATION
# ============================================================
def draw_ax(ax, perm, title, clr_main):
    CLRS=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c",
          "#e67e22","#34495e","#e91e63","#00bcd4","#ff5722","#607d8b","#795548"]
    routes=perm_to_sol(perm)
    xs=[c[0] for c in customers]; ys=[c[1] for c in customers]
    ax.scatter(xs,ys,c="#bdc3c7",s=18,zorder=2,linewidths=0)
    for ri,rt in enumerate(routes):
        c=CLRS[ri%len(CLRS)]
        path=[depot]+[(customers[v][0],customers[v][1]) for v in rt]+[depot]
        for k in range(len(path)-1):
            ax.plot([path[k][0],path[k+1][0]],[path[k][1],path[k+1][1]],
                    "-",color=c,linewidth=1.2,alpha=0.8,zorder=3)
        for v in rt: ax.scatter(customers[v][0],customers[v][1],c=c,s=22,zorder=4,linewidths=0)
    ax.scatter(depot[0],depot[1],c="#e74c3c",s=140,zorder=6,
               marker="*",edgecolors="#c0392b",linewidths=0.8)
    d=fitness(perm); nv=len(routes)
    ax.set_title(f"{title}\nDist:{d:.1f} | Xe:{nv}",
                 fontsize=10,fontweight="bold",color=clr_main,pad=6)
    ax.set_xlim(-5,105); ax.set_ylim(-5,100)
    ax.set_facecolor("#f8f9fa"); ax.grid(True,alpha=0.25,linewidth=0.5); ax.tick_params(labelsize=7)

def plot_dashboard(ga_p,pso_p,acs_p,ga_h,pso_h,acs_h,ga_t,pso_t,acs_t):
    ga_d=fitness(ga_p); pso_d=fitness(pso_p); acs_d=fitness(acs_p)
    ga_v=count_vehicles(ga_p); pso_v=count_vehicles(pso_p); acs_v=count_vehicles(acs_p)
    CLR={"GA":"#3498db","PSO":"#2ecc71","ACS":"#e67e22"}
    fig=plt.figure(figsize=(18,14),facecolor="#1a1a2e")
    fig.suptitle("VRPTW · GA / PSO / ACS · Solomon C101 · seed=42 · pop=100 · iter=500",
                 fontsize=13,fontweight="bold",color="white",y=0.98)
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.48,wspace=0.30,
                         top=0.93,bottom=0.04,left=0.05,right=0.97)
    for ci,(perm,nm,lb) in enumerate([(ga_p,"GA","Genetic Algorithm (GA)"),
                                       (pso_p,"PSO","Particle Swarm (PSO)"),
                                       (acs_p,"ACS","Ant Colony System (ACS)")]):
        draw_ax(fig.add_subplot(gs[0,ci]),perm,lb,CLR[nm])

    ac=fig.add_subplot(gs[1,:2]); ac.set_facecolor("#0f0f23")
    for h,nm in [(ga_h,"GA"),(pso_h,"PSO"),(acs_h,"ACS")]:
        ac.plot(range(len(h)),h,color=CLR[nm],linewidth=2.0,label=nm)
        ac.scatter(len(h)-1,h[-1],color=CLR[nm],s=60,zorder=5)
        ac.annotate(f"{h[-1]:.0f}",xy=(len(h)-1,h[-1]),xytext=(6,4),
                    textcoords="offset points",fontsize=8,color=CLR[nm],fontweight="bold")
    ac.set_title("Convergence",fontsize=11,fontweight="bold",color="white",pad=6)
    ac.set_xlabel("Iteration",fontsize=9,color="#aaa"); ac.set_ylabel("Best Distance",fontsize=9,color="#aaa")
    ac.tick_params(colors="#aaa",labelsize=8); ac.spines[:].set_color("#333")
    ac.legend(fontsize=9,facecolor="#1a1a2e",labelcolor="white",edgecolor="#444"); ac.grid(True,alpha=0.2,linewidth=0.5)

    ab=fig.add_subplot(gs[1,2]); ab.set_facecolor("#0f0f23")
    nms=["GA","PSO","ACS"]; dists=[ga_d,pso_d,acs_d]; paper=[1257.19,1830.69,1044.81]
    bi=dists.index(min(dists))
    bars=ab.bar(nms,dists,color=[CLR[n] for n in nms],width=0.4,edgecolor="#333")
    bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(2.5)
    for bar,val,pv in zip(bars,dists,paper):
        ab.text(bar.get_x()+bar.get_width()/2,bar.get_height()+12,
                f"{val:.0f}\n(báo:{pv:.0f})",ha="center",va="bottom",fontsize=8,color="white",fontweight="bold")
    ab.set_title("Result vs Paper",fontsize=11,fontweight="bold",color="white",pad=6)
    ab.set_ylabel("Distance",fontsize=9,color="#aaa"); ab.tick_params(colors="#aaa",labelsize=9)
    ab.spines[:].set_color("#333"); ab.grid(True,axis="y",alpha=0.2,linewidth=0.5)
    ab.set_ylim(min(dists)*0.88,max(dists)*1.14)

    at=fig.add_subplot(gs[2,:]); at.set_facecolor("#0f0f23"); at.axis("off")
    winner=nms[bi]; wc={"GA":1,"PSO":2,"ACS":3}[winner]
    rows=[["Thuật toán","GA","PSO","ACS"],
          ["Kết quả (1 run)",f"{ga_d:.2f}",f"{pso_d:.2f}",f"{acs_d:.2f}"],
          ["Bài báo (Avg×21)","1257.19","1830.69","1044.81"],
          ["Số xe",str(ga_v),str(pso_v),str(acs_v)],
          ["Thời gian (s)",f"{ga_t:.1f}",f"{pso_t:.1f}",f"{acs_t:.1f}"],
          ["Init method","NN heuristic","NN heuristic","NN heuristic"],
          ["Pop/Agents","100","100","100 ants"],
          ["Iterations","500 gen","500 iter","500 iter"]]
    hbg=["#2c3e50","#1a5276","#145a32","#784212"]
    dbg=["#1c2833","#1a2740","#0d2e1a","#3d2006"]
    cw=[0.22,0.26,0.26,0.26]; rh=0.11
    x0s=[0.02]
    for c in cw[:-1]: x0s.append(x0s[-1]+c)
    for ri,row in enumerate(rows):
        for ci,cell in enumerate(row):
            x0=x0s[ci]; y0=1.0-(ri+1)*rh-0.02
            hr=ri==0; hc=ci==0; wcc=ci==wc and not hc
            if hr:   bg,fc,fw,fs=hbg[ci],"white","bold",10
            elif hc: bg,fc,fw,fs="#2c3e50","#aab7b8","bold",9
            elif wcc and ri==1: bg,fc,fw,fs="#7d6608","gold","bold",10
            elif wcc: bg,fc,fw,fs=dbg[ci],"#f0f0a0","normal",9
            else:    bg,fc,fw,fs=dbg[ci],"#cccccc","normal",9
            rect=FancyBboxPatch((x0+0.002,y0+0.005),cw[ci]-0.005,rh-0.01,
                                boxstyle="round,pad=0.005",facecolor=bg,edgecolor="#444",
                                linewidth=0.6,transform=at.transAxes,clip_on=False)
            at.add_patch(rect)
            at.text(x0+cw[ci]/2,y0+rh/2,cell,ha="center",va="center",
                    fontsize=fs,color=fc,fontweight=fw,transform=at.transAxes)
    at.text(x0s[wc]+cw[wc]/2,1.0-0.5*rh+0.02,"🏆 WINNER",ha="center",va="center",
            fontsize=9,color="gold",fontweight="bold",transform=at.transAxes)
    at.set_title("Bảng so sánh (1 run seed=42  vs  trung bình 21 lần trong bài báo)",
                 fontsize=11,fontweight="bold",color="white",pad=8)
    out="dashboard_VRPTW_fixed.png"
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    print(f"\n>>> Saved: {out}"); plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":
    print("="*62)
    print("  Solomon C101 | SEED=42 → kết quả GIỐNG NHAU mỗi lần chạy")
    print("  Paper avg (21 runs): GA=1257.19  PSO=1830.69  ACS=1044.81")
    print("="*62)
    ga_p,  ga_h,  ga_t  = run_GA( pop_size=100, generations=500, seed=SEED)
    pso_p, pso_h, pso_t = run_PSO(pop_size=100, max_iter=500,    seed=SEED)
    acs_p, acs_h, acs_t = run_ACS(n_ants=100,   n_iter=500,      seed=SEED)

    print("\n"+"="*62+"  TỔNG KẾT")
    print(f"  {'Algo':<6} {'This run':>10} {'Paper Avg':>12} {'Veh':>5} {'Time(s)':>9}")
    print("  "+"-"*47)
    for nm,perm,pv,t in [("GA",ga_p,1257.19,ga_t),("PSO",pso_p,1830.69,pso_t),("ACS",acs_p,1044.81,acs_t)]:
        d=fitness(perm); v=count_vehicles(perm)
        diff=d-pv; sign="+" if diff>=0 else ""
        print(f"  {nm:<6} {d:>10.2f} {pv:>12.2f} {v:>5} {t:>9.1f}s   (diff={sign}{diff:.2f})")
    print("\n  * SEED=42 cố định → chạy lại 100 lần kết quả vẫn GIỐNG NHAU.")
    print("  * Bài báo dùng trung bình 21 runs ngẫu nhiên; đây là 1 run có seed.")
    plot_dashboard(ga_p,pso_p,acs_p,ga_h,pso_h,acs_h,ga_t,pso_t,acs_t)