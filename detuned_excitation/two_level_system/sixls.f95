! compile with 'f2py -c sixls.f95 -m sixls'
subroutine sixls_rk4(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, polar_p, polar_m, tau_0,&
    e_energy, alpha, e0, bx, bz, delta_b, d0, d1, d2, delta_E)
    ! ===============================
    ! solves six level system for one pulse with given parameters
    ! tau_0, e_energy, alpha, e0 are pulse parameters
    ! rotating frame with energy of E_bm is used. 
    ! be careful if two consecutive pulses with different Bz are used
    ! if so, use different phidot which does not depend on Bz
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, delta_E
    Real*8,intent(in) :: dt, polar_p, polar_m, tau_0, e_energy, alpha, e0, bx, bz, delta_B, d0, d1, d2
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(6)
    Complex*16,intent(in):: in_polar(15)
    Real*8,intent(out):: out_state(6)
    Complex*16,intent(out):: out_polar(15)
    Real*8,intent(out):: out_states(n_steps,6)
    Complex*16,intent(out):: out_polars(n_steps,15)
    Real*8 :: HBAR, g_ex, g_ez, g_hx, g_hz, mu_b, gex, ghx
    Real*8 :: E_dp, E_bp, E_bm, E_dm, E_b
    Real*8 :: t, tau, a_chirp, pi, w_start, phidot, phi
    Real*8 :: k1r(6), k2r(6), k3r(6), k4r(6), energies(5)
    Complex*16 :: k1i(15), k2i(15), k3i(15), k4i(15), omm, omp
    Complex*16 :: ii = (0.0,1.0)
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! d0 = 0.25 !250E-03  ! meV
    ! d1 = 0.12 !120E-03
    ! d2 = 0.05 !50E-3
    out_state = in_state
    g_ex = -0.65
    g_ez = -0.8
    g_hx = -0.35
    g_hz = -2.2
    mu_b = 5.7882818012E-02  ! meV/T
    ! delta_B = -d0
    !bz = 4.0
    !bx = 3.0
    ! starting parameters
    !g=in_state(1); dp=in_state(2); bp=in_state(3); bm=in_state(4); dm=in_state(5); b=in_state(6);
    !gdm=in_polar(1); gbm=in_polar(2); gbp=in_polar(3); gdp=in_polar(4); gb=in_polar(5); dpbp=in_polar(6); dpbm=in_polar(7);
    !dpdm=in_polar(8);dpb=in_polar(9); bpbm=in_polar(10); bpdm=in_polar(11); bpb=in_polar(12);
    !bmdm=in_polar(13); bmb=in_polar(14); dmb=in_polar(15);
    ! energies
    E_dp = delta_E - d0/2.0 - mu_b*bz/2. * ( 3.*g_hz + g_ez)
    E_bp = delta_E + d0/2.0 + mu_b*bz/2. * (-3.*g_hz + g_ez)
    E_bm = delta_E + d0/2.0 - mu_b*bz/2. * (-3.*g_hz + g_ez)
    E_dm = delta_E - d0/2.0 + mu_b*bz/2. * ( 3.*g_hz + g_ez)
    E_b = 2.*delta_E - delta_B
    ! pulse parameters
    tau = sqrt((alpha**2. / tau_0**2.) + tau_0**2. )
    a_chirp = alpha / (alpha**2. + tau_0**4.)
    w_start = e_energy / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    gex = g_ex * mu_b * bx
    ghx = g_hx * mu_b * bx

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_bm 
    ! actually phi/t. if this is time dependent, move inside the loop
    phi = E_bm / HBAR
    energies(1)=E_dp; energies(2)=E_bp; energies(3)=E_bm; energies(4)=E_dm; energies(5)=E_b
    do i = 0, n_steps - 2

        ! take first rk4 step:
        t = t_0 + i * dt
        omp = polar_p * e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-phi + 0.5*a_chirp*t)*t) / sqrt(2.*pi*tau * tau_0) 
        omm = polar_m * e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-phi + 0.5*a_chirp*t)*t) / sqrt(2.*pi*tau * tau_0) 

        
        
        call sixls_eq_rf(out_states(i+1,:), out_polars(i+1,:), phidot, omm, omp, energies, d1, d2, gex, ghx, k1r, k1i)

        ! now t -> t+h/2
        t = t_0 + i * dt + 0.5*dt
        omp = polar_p * e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-phi + 0.5*a_chirp*t)*t) / sqrt(2.*pi*tau * tau_0) 
        omm = polar_m * e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-phi + 0.5*a_chirp*t)*t) / sqrt(2.*pi*tau * tau_0) 
        
        call sixls_eq_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, omm, omp, energies, d1, d2, gex, ghx,&
                         k2r, k2i)
        call sixls_eq_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot,  omm, omp, energies, d1, d2, gex, ghx,&
                         k3r, k3i)
        call sixls_eq_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, omm, omp, energies, d1, d2, gex, ghx, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)

end subroutine sixls_rk4


subroutine sixls_twopulse(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, polar_m1, polar_m2, &
    tau1, tau2, e_energy1, e_energy2, e01, e02, bx, bz, delta_b, d0, d1, d2, delta_E, t02)
    ! ===============================
    ! solves six level system for one pulse with given parameters
    ! tau_0, e_energy, alpha, e0 are pulse parameters
    ! rotating frame with energy of E_bm is used. 
    ! be careful if two consecutive pulses with different Bz are used
    ! if so, use different phidot which does not depend on Bz
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, delta_E
    Real*8,intent(in) :: dt, polar_m1, polar_m2, tau1, tau2, e_energy1, e_energy2, e01, e02, bx, bz, delta_B, d0, d1, d2, t02
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(6)
    Complex*16,intent(in):: in_polar(15)
    Real*8,intent(out):: out_state(6)
    Complex*16,intent(out):: out_polar(15)
    Real*8,intent(out):: out_states(n_steps,6)
    Complex*16,intent(out):: out_polars(n_steps,15)
    Real*8 :: HBAR, g_ex, g_ez, g_hx, g_hz, mu_b, gex, ghx
    Real*8 :: E_dp, E_bp, E_bm, E_dm, E_b
    Real*8 :: t, pi, w_1, w_2, phidot, phi, pm1, pm2, pp1, pp2
    Real*8 :: k1r(6), k2r(6), k3r(6), k4r(6), energies(5)
    Complex*16 :: k1i(15), k2i(15), k3i(15), k4i(15), omm, omp, laser1, laser2
    Complex*16 :: ii = (0.0,1.0)
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! d0 = 0.25 !250E-03  ! meV
    ! d1 = 0.12 !120E-03
    ! d2 = 0.05 !50E-3
    out_state = in_state
    g_ex = -0.65
    g_ez = -0.8
    g_hx = -0.35
    g_hz = -2.2
    mu_b = 5.7882818012E-02  ! meV/T
    ! delta_B = -d0
    !bz = 4.0
    !bx = 3.0
    ! starting parameters
    !g=in_state(1); dp=in_state(2); bp=in_state(3); bm=in_state(4); dm=in_state(5); b=in_state(6);
    !gdm=in_polar(1); gbm=in_polar(2); gbp=in_polar(3); gdp=in_polar(4); gb=in_polar(5); dpbp=in_polar(6); dpbm=in_polar(7);
    !dpdm=in_polar(8);dpb=in_polar(9); bpbm=in_polar(10); bpdm=in_polar(11); bpb=in_polar(12);
    !bmdm=in_polar(13); bmb=in_polar(14); dmb=in_polar(15);
    ! energies
    E_dp = delta_E - d0/2.0 - mu_b*bz/2. * ( 3.*g_hz + g_ez)
    E_bp = delta_E + d0/2.0 + mu_b*bz/2. * (-3.*g_hz + g_ez)
    E_bm = delta_E + d0/2.0 - mu_b*bz/2. * (-3.*g_hz + g_ez)
    E_dm = delta_E - d0/2.0 + mu_b*bz/2. * ( 3.*g_hz + g_ez)
    E_b = 2.*delta_E - delta_B
    ! pulse parameters
    !tau = sqrt((alpha**2. / tau_0**2.) + tau_0**2. )
    !a_chirp = alpha / (alpha**2. + tau_0**4.)
    w_1 = e_energy1 / HBAR
    w_2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    gex = g_ex * mu_b * bx
    ghx = g_hx * mu_b * bx

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    ! laser polarizations
    pm1 = polar_m1
    pm2 = polar_m2
    pp1 = sqrt(1 - pm1**2)
    pp2 = sqrt(1 - pm2**2)

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_bm 
    ! actually phi/t. if this is time dependent, move inside the loop
    phi = E_bm / HBAR
    energies(1)=E_dp; energies(2)=E_bp; energies(3)=E_bm; energies(4)=E_dm; energies(5)=E_b
    do i = 0, n_steps - 2

        ! take first rk4 step:
        t = t_0 + i * dt
        laser1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_1-phi)*t)/ sqrt(2.*pi*tau1 * tau1)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*(w_2-phi)*(t-t02))/ sqrt(2.*pi*tau2 * tau2)
        omp = pp1*laser1 + pp2*laser2
        omm = pm1*laser1 + pm2*laser2
        
        
        call sixls_eq_rf(out_states(i+1,:), out_polars(i+1,:), phidot, omm, omp, energies, d1, d2, gex, ghx, k1r, k1i)

        ! now t -> t+h/2
        t = t_0 + i * dt + 0.5*dt
        laser1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_1-phi)*t)/ sqrt(2.*pi*tau1 * tau1)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*(w_2-phi)*(t-t02))/ sqrt(2.*pi*tau2 * tau2)
        omp = pp1*laser1 + pp2*laser2
        omm = pm1*laser1 + pm2*laser2
        
        call sixls_eq_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, omm, omp, energies, d1, d2, gex, ghx,&
                         k2r, k2i)
        call sixls_eq_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot,  omm, omp, energies, d1, d2, gex, ghx,&
                         k3r, k3i)
        call sixls_eq_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, omm, omp, energies, d1, d2, gex, ghx, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)

end subroutine sixls_twopulse


subroutine sixls_eq_rf(in_state, in_polar, phidot, omm, omp, energies, d1, d2, gex, ghx, out_state, out_polar)
    implicit none
    ! equations for six level system in a rotating frame. 
    ! electric fields can be complex to include a phase
    ! or an oscillation
    Real*8,intent(in) :: in_state(6)
    Complex*16,intent(in) :: in_polar(15), omm, omp
    Real*8,intent(in) :: energies(5), d1, d2, gex, ghx, phidot
    Real*8,intent(out) :: out_state(6)
    Complex*16,intent(out) :: out_polar(15)
    Real*8 :: HBAR, E_dp, E_bp, E_bm, E_dm, E_b
    Complex*16 :: ii = (0.0,1.0)
    Real*8 :: g, dp, bp, bm, dm, b
    Complex*16 :: gdm, gbm, gbp, gdp, gb, dpbp, dpbm, dpdm, dpb, bpbm, bpdm, bpb, bmdm, bmb, dmb
    
    HBAR = 6.582119514E02  ! meV fs
    E_dp=energies(1); E_bp=energies(2); E_bm=energies(3); E_dm=energies(4); E_b=energies(5)
    g=in_state(1); dp=in_state(2); bp=in_state(3); bm=in_state(4); dm=in_state(5); b=in_state(6);
    gdm=in_polar(1); gbm=in_polar(2); gbp=in_polar(3); gdp=in_polar(4); gb=in_polar(5); dpbp=in_polar(6); dpbm=in_polar(7);
    dpdm=in_polar(8);dpb=in_polar(9); bpbm=in_polar(10); bpdm=in_polar(11); bpb=in_polar(12);
    bmdm=in_polar(13); bmb=in_polar(14); dmb=in_polar(15);

    out_state(1) = real((0.5)*ii*gbm*conjg(omm) + (0.5)*ii*gbp*conjg(omp) - 0.5*ii*omm*conjg(gbm) - 0.5*ii*omp*conjg(gbp))
    out_polar(4) = -0.5*ii*omm*conjg(dpbm) - 0.5*ii*omp*conjg(dpbp) - 0.5*ii*d2*gdm/HBAR + 0.5*ii*gbm*ghx/HBAR &
    + 0.5*ii*gbp*gex/HBAR + ii*gdp*(-E_dp + phidot)/HBAR
    out_polar(3) = -0.5*ii*bp*omp + (0.5)*ii*g*omp + (0.5)*ii*gb*conjg(omm) - 0.5*ii*omm*conjg(bpbm) - 0.5*ii*d1*gbm/HBAR &
    + ii*gbp*(-E_bp + phidot)/HBAR + 0.5*ii*gdm*ghx/HBAR + 0.5*ii*gdp*gex/HBAR
    out_polar(2) = -0.5*ii*bm*omm - 0.5*ii*bpbm*omp + (0.5)*ii*g*omm + (0.5)*ii*gb*conjg(omp) - 0.5*ii*d1*gbp/HBAR &
    + ii*gbm*(-E_bm + phidot)/HBAR + 0.5*ii*gdm*gex/HBAR + 0.5*ii*gdp*ghx/HBAR
    out_polar(1) = -0.5*ii*bmdm*omm - 0.5*ii*bpdm*omp - 0.5*ii*d2*gdp/HBAR + 0.5*ii*gbm*gex/HBAR + 0.5*ii*gbp*ghx/HBAR &
    + ii*gdm*(-E_dm + phidot)/HBAR
    out_polar(5) = -0.5*ii*bmb*omm - 0.5*ii*bpb*omp + (0.5)*ii*gbm*omp + (0.5)*ii*gbp*omm + ii*gb*(-E_b + 2*phidot)/HBAR
    out_state(2) = real(-0.5*ii*d2*dpdm/HBAR + 0.5*ii*d2*conjg(dpdm)/HBAR + 0.5*ii*dpbm*ghx/HBAR + 0.5*ii*dpbp*gex/HBAR &
    - 0.5*ii*gex*conjg(dpbp)/HBAR - 0.5*ii*ghx*conjg(dpbm)/HBAR)
    out_polar(6) = (0.5)*ii*dpb*conjg(omm) + (0.5)*ii*omp*conjg(gdp) - 0.5*ii*bp*gex/HBAR - 0.5*ii*d1*dpbm/HBAR &
    + 0.5*ii*d2*conjg(bpdm)/HBAR + 0.5*ii*dp*gex/HBAR + ii*dpbp*(-E_bp + E_dp)/HBAR + 0.5*ii*dpdm*ghx/HBAR &
    - 0.5*ii*ghx*conjg(bpbm)/HBAR
    out_polar(7) = (0.5)*ii*dpb*conjg(omp) + (0.5)*ii*omm*conjg(gdp) - 0.5*ii*bm*ghx/HBAR - 0.5*ii*bpbm*gex/HBAR &
    - 0.5*ii*d1*dpbp/HBAR + 0.5*ii*d2*conjg(bmdm)/HBAR + 0.5*ii*dp*ghx/HBAR + ii*dpbm*(-E_bm + E_dp)/HBAR &
    + 0.5*ii*dpdm*gex/HBAR
    out_polar(8) = -0.5*ii*bmdm*ghx/HBAR - 0.5*ii*bpdm*gex/HBAR + 0.5*ii*d2*dm/HBAR - 0.5*ii*d2*dp/HBAR + 0.5*ii*dpbm*gex/HBAR &
    + 0.5*ii*dpbp*ghx/HBAR + ii*dpdm*(-E_dm + E_dp)/HBAR
    out_polar(9) = (0.5)*ii*dpbm*omp + (0.5)*ii*dpbp*omm - 0.5*ii*bmb*ghx/HBAR - 0.5*ii*bpb*gex/HBAR + 0.5*ii*d2*dmb/HBAR &
    + ii*dpb*(-E_b + E_dp + phidot)/HBAR
    out_state(3) = real((0.5)*ii*bpb*conjg(omm) - 0.5*ii*gbp*conjg(omp) - 0.5*ii*omm*conjg(bpb) + (0.5)*ii*omp*conjg(gbp) &
    - 0.5*ii*bpbm*d1/HBAR + 0.5*ii*bpdm*ghx/HBAR + 0.5*ii*d1*conjg(bpbm)/HBAR - 0.5*ii*dpbp*gex/HBAR &
    + 0.5*ii*gex*conjg(dpbp)/HBAR - 0.5*ii*ghx*conjg(bpdm)/HBAR)
    out_polar(10) = (0.5)*ii*bpb*conjg(omp) - 0.5*ii*gbm*conjg(omp) - 0.5*ii*omm*conjg(bmb) + (0.5)*ii*omm*conjg(gbp) &
    + 0.5*ii*bm*d1/HBAR - 0.5*ii*bp*d1/HBAR + ii*bpbm*(-E_bm + E_bp)/HBAR + 0.5*ii*bpdm*gex/HBAR &
    - 0.5*ii*dpbm*gex/HBAR - 0.5*ii*ghx*conjg(bmdm)/HBAR + 0.5*ii*ghx*conjg(dpbp)/HBAR
    out_polar(11) = -0.5*ii*gdm*conjg(omp) - 0.5*ii*omm*conjg(dmb) + 0.5*ii*bmdm*d1/HBAR + 0.5*ii*bp*ghx/HBAR &
    + 0.5*ii*bpbm*gex/HBAR + ii*bpdm*(E_bp - E_dm)/HBAR - 0.5*ii*d2*conjg(dpbp)/HBAR - 0.5*ii*dm*ghx/HBAR &
    - 0.5*ii*dpdm*gex/HBAR
    out_polar(12) = -0.5*ii*b*omm + (0.5)*ii*bp*omm + (0.5)*ii*bpbm*omp - 0.5*ii*gb*conjg(omp) + 0.5*ii*bmb*d1/HBAR &
    + ii*bpb*(-E_b + E_bp + phidot)/HBAR - 0.5*ii*dmb*ghx/HBAR - 0.5*ii*dpb*gex/HBAR
    out_state(4) = real((0.5)*ii*bmb*conjg(omp) - 0.5*ii*gbm*conjg(omm) + (0.5)*ii*omm*conjg(gbm) - 0.5*ii*omp*conjg(bmb) &
    + 0.5*ii*bmdm*gex/HBAR + 0.5*ii*bpbm*d1/HBAR - 0.5*ii*d1*conjg(bpbm)/HBAR - 0.5*ii*dpbm*ghx/HBAR &
    - 0.5*ii*gex*conjg(bmdm)/HBAR + 0.5*ii*ghx*conjg(dpbm)/HBAR)
    out_polar(13) = -0.5*ii*gdm*conjg(omm) - 0.5*ii*omp*conjg(dmb) + 0.5*ii*bm*gex/HBAR + ii*bmdm*(E_bm - E_dm)/HBAR &
    + 0.5*ii*bpdm*d1/HBAR - 0.5*ii*d2*conjg(dpbm)/HBAR - 0.5*ii*dm*gex/HBAR - 0.5*ii*dpdm*ghx/HBAR &
    + 0.5*ii*ghx*conjg(bpbm)/HBAR
    out_polar(14) = -0.5*ii*b*omp + (0.5)*ii*bm*omp - 0.5*ii*gb*conjg(omm) + (0.5)*ii*omm*conjg(bpbm) &
    + ii*bmb*(-E_b + E_bm + phidot)/HBAR + 0.5*ii*bpb*d1/HBAR - 0.5*ii*dmb*gex/HBAR - 0.5*ii*dpb*ghx/HBAR
    out_state(5) = real(-0.5*ii*bmdm*gex/HBAR - 0.5*ii*bpdm*ghx/HBAR + 0.5*ii*d2*dpdm/HBAR - 0.5*ii*d2*conjg(dpdm)/HBAR &
    + 0.5*ii*gex*conjg(bmdm)/HBAR + 0.5*ii*ghx*conjg(bpdm)/HBAR)
    out_polar(15) = (0.5)*ii*omm*conjg(bpdm) + (0.5)*ii*omp*conjg(bmdm) - 0.5*ii*bmb*gex/HBAR - 0.5*ii*bpb*ghx/HBAR &
    + 0.5*ii*d2*dpb/HBAR + ii*dmb*(-E_b + E_dm + phidot)/HBAR
    out_state(6) = real(-0.5*ii*bmb*conjg(omp) - 0.5*ii*bpb*conjg(omm) + (0.5)*ii*omm*conjg(bpb) + (0.5)*ii*omp*conjg(bmb))

end subroutine sixls_eq_rf
