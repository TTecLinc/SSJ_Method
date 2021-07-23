
warning off;

options_.maxit_=1000000;
var y, c, I, k, g, n, v , k_hat, PI, PI_tilde, A, D, mc, w, R, lambda, mu, i, deta, dy, A_hat, D_hat, dI, dc, dR, Z, dn, u_c, u_n, deta_hat, U, L;

//y is output, c is consumption, I is investment, k is capital, g is government, n is labor, 
//v is the distortion of relative prices, k_hat is utilized capital (I've set this to equal capital)
//PI is inflation +1, PI_tilde is reset inflation +1, A and D are used for recursive sums in optimal
//price setting, mc is marginal cost, w is wage, R is return to capital, lambda is the lagrange multiplier
//on the budget constraint, mu is the lagrange multiplier on the capital accumulation constraint, 
//i is the interest rate, deta is the discount factor, xc is a persistent shock (unused)

varexo e_i exc exi;

//exc is the discount factor shock

parameters Lss, tau_n, tau_c, tau_k, rho_r,sigma, sigma_I,beta,gamma,alpha, theta, epsilon, DELTA, rho_g, phi_pi, phi_y, omega, phi, delta,nss,Rss,mcss,
    wss, k_hatss,  epsilon_w,  kss,    css, yss, iss, lambdass, muss, Iss, uss, vss, PIss, PI_tildess, Ass, Dss, gss, qss, rho_phi, rc;

beta = 0.99; //discount factor
alpha = 0.3; //capital share
omega = 0.2; //ratio of govt expenditure
delta = 0.02; //depreciation 
gamma =0.29; //utility parameter
sigma = 2; //risk aversion
epsilon = 7; //intermediate good substitubility
sigma_I = 17; //adjustment parameter
rho_g = 0.8; //presistence in govt
phi_pi=1.5; //taylor for inflation
phi_y = 0.0; //taylor for output
theta = 0.85; //calvo
DELTA = (1/beta-1)/delta +1; //unused for now
rho_r = 0;
tau_n =  0.28;
tau_k = 0.36;
tau_c= 0.05;
epsilon_w = 3;

//shocks
rc =0.95; //persistence parameter on unused shock
// Benchmark calibration and steady state 
    nss = 0.33; //labor
    Rss = 1/beta-(1-delta);
    mcss = (epsilon-1)/epsilon;
    wss = mcss*(1-alpha)*(alpha*mcss/Rss)^(alpha/(1-alpha));
    k_hatss = ((alpha*mcss)/Rss)^(1/(1-alpha))*nss;
    kss = k_hatss;
    css = ((1-omega)*(kss/nss)^alpha -delta*kss/nss)*nss;
    yss = (kss/nss)^alpha*nss;
    iss = 1/beta-1;
    lambdass = (css^gamma*(1-nss)^(1-gamma))^(-sigma)*gamma*((1-nss)/css)^(1-gamma);
    muss= beta*lambdass*Rss/(1+beta*(delta-1));
    Iss = delta*kss;
    uss = 1;
    vss =1;
    PIss = 1;
    PI_tildess = 1;
    Ass = lambdass*yss*mcss/(1-beta*theta);
    Dss = lambdass*yss/(1-beta*theta);
    gss = omega*yss;
    qss = 1;
    Lss = 1-nss;
    


model;
    L = 1-n;
    deta_hat = deta(-1);
    U = (c^gamma*L^(1-gamma))^(1-sigma)/(1-sigma) + deta_hat(+1)*U(+1);
    deta = beta + exc; //shock to the discount factor
    y = c+ I +g; //aggregate accounting identity
    y = (k_hat)^alpha*n^(1-alpha)/v; // aggregate production function
    k_hat = k; //capital utilization
    PI = ((1-theta)*(PI_tilde)^(1-epsilon)+theta)^(1/(1-epsilon)); //evolution of aggregate inflation
    v = (1-theta)*(PI/PI_tilde)^(epsilon)+theta*PI^(epsilon)*v(-1);//aggregate price dispresion  ??? - epsilon ???
    A = lambda*y*mc+theta*deta*((PI(+1))^epsilon)*A(+1);//auxiliary terms
    D = lambda*y + theta*deta*((PI(+1))^(epsilon-1))*D(+1);//auxiliary terms 
    A_hat = A(+1);
    D_hat = D(+1);
    PI_tilde = PI*(epsilon/(epsilon-1))*A_hat/D_hat;//reset inflation evolution
    w = mc*(1-alpha)*(k_hat/n)^alpha; //labor demand
    R = mc*alpha*(k_hat/n)^(alpha-1);//capital demand
    lambda*(1+tau_c) = u_c;//marginal utility of income
    u_c = (c^gamma*(1-n)^(1-gamma))^(-sigma)*gamma*((1-n)/c)^(1-gamma);
    u_n = (c^gamma*(1-n)^(1-gamma))^(-sigma)*(1-gamma)*(c/(1-n))^gamma;
    epsilon_w/(epsilon_w-1)*u_n = lambda*w*(1-tau_n);
    lambda = deta*lambda(+1)*(1+i(+1))*(PI(+1))^(-1);//Euler equation for bonds ??? i(+1) or just i ???
    mu = deta*(lambda(+1)*(R(+1)-tau_k*(R(+1)-delta))+mu(+1)*(1-delta-sigma_I/2*(I(+1)/k(+1)-delta)^2+sigma_I*(I(+1)/k(+1)-delta)*(I(+1)/k(+1)))); //FOC for capital
    lambda = mu*(1-sigma_I*(I/k-delta)); //FOC for investment
    k = I(-1)+(1-delta)*k(-1)-sigma_I/2*(I(-1)/k(-1)-delta)^2*k(-1); //capital accumulation
    log(g) = (1-rho_g)*(log(omega)+log(STEADY_STATE(y)))+rho_g*log(g(-1)); //government spending
    dy = (y/STEADY_STATE(y))-1;
    dc = (c/STEADY_STATE(c))-1;
    dR= (R/STEADY_STATE(R))-1;
    dn = (n/STEADY_STATE(n))-1;
    dI = (I/STEADY_STATE(I))-1;
    // Policies

    i = max(Z,0); //Central bank policy
   Z = 1/beta*(PI(-1))^(phi_pi*(1-rho_r))*(y(-1)/STEADY_STATE(y))^(phi_y*(1-rho_r))*(beta*(1+i(-1)))^(rho_r)-1;
end;

initval;
    y   = yss;
    c = css;
    I = Iss;
    g = gss;
    k_hat = kss;
    n = nss;
    v =vss;
    PI = 1;
    PI_tilde = 1;
    A = Ass;
    D = Dss;
    mc = mcss;
    w = wss;
    R = Rss;
    lambda = lambdass;
    mu = muss;
    i = iss;
    Z = iss;
    k = kss;
    deta = beta;
    A_hat = Ass;
    D_hat = Dss;
    deta_hat = deta;
    L = Lss;
    U = ((css^gamma*Lss^(1-gamma))^(1-sigma)/(1-sigma))/(1-beta);

end;
options_.slowc=0.00005; % you may lower this parameter if you want to stabilize the algorithm
resid
steady(solve_algo=2);
check;


%shock the discount factor by 0.02 for 1 period, simulate and plot the results. 
shocks;
    var exc;
    periods 1:10;
    values 0.02;
end;
simul(periods=400);

nrep    = 20;
smpl    = 2+(1:nrep);

close all

subplot(331);plot(4*Z(smpl),'LineWidth',2);title('Z')
subplot(332);plot(dI(smpl),'LineWidth',2);title('Investment')
subplot(333);plot(dn(smpl),'LineWidth',2);title('Hours')
subplot(334);plot(4*(PI(smpl)-1),'LineWidth',2);title('Inflation')
subplot(335);plot(4*(i(smpl)),'LineWidth',2);title('Nominal Interest Rate')
subplot(336);plot(dy(smpl),'LineWidth',2);title('Output')
subplot(337);plot(dc(smpl),'LineWidth',2);title('Consumption')
subplot(338);plot(4*(deta(smpl)-beta),'LineWidth',2);title('Discount Factor Shock')
subplot(339);plot(4*(i(smpl)-(PI(smpl)-1)),'LineWidth',2);title('Real Interest Rate')
