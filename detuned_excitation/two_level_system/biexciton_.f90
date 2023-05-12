
subroutine biex_twopulse(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, &
    tau1, tau2, e_energy1, e_energy2, e01, e02, delta_b, delta_E, t02, phase, alpha)
    ! ===============================
    ! solves biexciton / three level system for two pulses with given parameters
    ! rotating frame with energy of E_x is used. 
    ! the first pulse can be chirped by specifying alpha
    ! if big detunings are used, think about using one of those as rot. frame frequencies
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, delta_E, phase, alpha
    ! times are given in fs, energies in meV
    ! e01, e02 are pulse areas
    Real*8,intent(in) :: dt, tau1, tau2, e_energy1, e_energy2, e01, e02, delta_B, t02
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(3)
    Complex*16,intent(in):: in_polar(3)
    Real*8,intent(out):: out_state(3)
    Complex*16,intent(out):: out_polar(3)
    Real*8,intent(out):: out_states(n_steps,3)
    Complex*16,intent(out):: out_polars(n_steps,3)
    Real*8 :: HBAR, tau1_new, a_chirp
    Real*8 :: E_x, E_b
    Real*8 :: t, pi, w_1, w_2, phidot, phi
    Real*8 :: k1r(3), k2r(3), k3r(3), k4r(3), energies(2)
    Complex*16 :: k1i(3), k2i(3), k3i(3), k4i(3), omm, laser1, laser2
    Complex*16 :: ii = (0.0,1.0)
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    out_state = in_state
    
    !g=in_state(1); x=in_state(2); b=in_state(3)
    !gx=in_polar(1); gb=in_polar(2); xb=in_polar(3);
    ! energies
    E_x = delta_E
    E_b = 2.*delta_E - delta_B
    ! pulse parameters
    tau1_new = sqrt((alpha**2. / tau1**2.) + tau1**2. )
    a_chirp = alpha / (alpha**2. + tau1**4.)
    w_1 = e_energy1 / HBAR
    w_2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_x 
    ! actually phi/t. if this is time dependent (i.e. modulated frequency and using a time dependent rot. frame), move inside the loop
    phi = E_x / HBAR
    energies(1)=E_x; energies(2)=E_b
    do i = 0, n_steps - 2
        ! take first rk4 step:
        t = t_0 + i * dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omm = laser1 + laser2
        call biex_eq_rf(out_states(i+1,:), out_polars(i+1,:), phidot, omm, energies, k1r, k1i)

        ! now t -> t+h/2
        t = t_0 + i * dt + 0.5*dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omm = laser1 + laser2
        
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, omm, energies, k2r, k2i)
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot,  omm, energies, k3r, k3i)
        ! now t -> t+h
        t = t_0 + i * dt + dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omm = laser1 + laser2
        call biex_eq_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, omm, energies, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)

end subroutine biex_twopulse


subroutine biex_rectangle(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, &
    tau, e_energy1, e_energy2, e01, e02, delta_b, delta_E)
    ! ===============================
    ! solves biexciton / three level system for two rectangle pulses with given parameters
    ! rotating frame with energy of Ex is used.
    ! both pulses have the same duration tau. 
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, delta_E
    ! times are given in fs, energies in meV
    ! e01, e02 are pulse areas
    Real*8,intent(in) :: dt, tau, e_energy1, e_energy2, e01, e02, delta_B
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(3)
    Complex*16,intent(in):: in_polar(3)
    Real*8,intent(out):: out_state(3)
    Complex*16,intent(out):: out_polar(3)
    Real*8,intent(out):: out_states(n_steps,3)
    Complex*16,intent(out):: out_polars(n_steps,3)
    Real*8 :: HBAR
    Real*8 :: E_x, E_b
    Real*8 :: t, pi, w_1, w_2, phidot, phi
    Real*8 :: k1r(3), k2r(3), k3r(3), k4r(3), energies(2)
    Complex*16 :: k1i(3), k2i(3), k3i(3), k4i(3), omm, laser1, laser2
    Complex*16 :: ii = (0.0,1.0)
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    out_state = in_state
    
    !g=in_state(1); x=in_state(2); b=in_state(3)
    !gx=in_polar(1); gb=in_polar(2); xb=in_polar(3);
    ! energies
    E_x = delta_E
    E_b = 2.*delta_E - delta_B
    ! pulse parameters
    w_1 = e_energy1 / HBAR
    w_2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_x 
    ! actually phi/t. if this is time dependent (i.e. modulated frequency and using a time dependent rot. frame), move inside the loop
    phi = E_x / HBAR
    energies(1)=E_x; energies(2)=E_b
    do i = 0, n_steps - 2
        ! take first rk4 step:
        t = t_0 + i * dt
        laser1 = 0
        laser2 = 0
        if (abs(t) < tau/2.0 ) then
            laser1 = e01/tau * exp(-ii*(w_1-phi)*t)
            laser2 = e02/tau * exp(-ii*(w_2-phi)*t)
        end if
        omm = laser1 + laser2
        call biex_eq_rf(out_states(i+1,:), out_polars(i+1,:), phidot, omm, energies, k1r, k1i)

        ! now t -> t+h/2
        t = t_0 + i * dt + 0.5*dt
        laser1 = 0
        laser2 = 0
        if (abs(t) < tau/2.0 ) then
            laser1 = e01/tau * exp(-ii*(w_1-phi)*t)
            laser2 = e02/tau * exp(-ii*(w_2-phi)*t)
        end if
        omm = laser1 + laser2
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, omm, energies, k2r, k2i)
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot,  omm, energies, k3r, k3i)
        
        ! now t -> t+h
        t = t_0 + i * dt + dt
        laser1 = 0
        laser2 = 0
        if (abs(t) < tau/2.0 ) then
            laser1 = e01/tau * exp(-ii*(w_1-phi)*t)
            laser2 = e02/tau * exp(-ii*(w_2-phi)*t)
        end if
        omm = laser1 + laser2
        call biex_eq_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, omm, energies, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)

end subroutine biex_rectangle

subroutine biex_linear_twopulse(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, &
    tau1, tau2, e_energy1, e_energy2, e01, e02, pol1x, pol2x, delta_b, delta_0, delta_E, t02, phase, alpha)
    ! ===============================
    ! solves biexciton / three level system for two pulses with given parameters
    ! rotating frame with energy of E_x is used. 
    ! the first pulse can be chirped by specifying alpha
    ! if big detunings are used, think about using one of those as rot. frame frequencies
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, delta_E, phase, alpha
    ! times are given in fs, energies in meV
    ! e01, e02 are pulse areas
    Real*8,intent(in) :: dt, tau1, tau2, e_energy1, e_energy2, e01, e02, delta_B, t02, pol1x, pol2x, delta_0
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(4)
    Complex*16,intent(in):: in_polar(6)
    Real*8,intent(out):: out_state(4)
    Complex*16,intent(out):: out_polar(6)
    Real*8,intent(out):: out_states(n_steps,4)
    Complex*16,intent(out):: out_polars(n_steps,6)
    Real*8 :: HBAR, tau1_new, a_chirp
    Real*8 :: E_x, E_y, E_b
    Real*8 :: t, pi, w_1, w_2, phidot, phi, pol1y, pol2y
    Real*8 :: k1r(4), k2r(4), k3r(4), k4r(4), energies(3)
    Complex*16 :: k1i(6), k2i(6), k3i(6), k4i(6), omx, omy, laser1, laser2
    Complex*16 :: ii = (0.0,1.0)
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 0.6582119514  ! meV ps
    out_state = in_state
    
    !g=in_state(1); x=in_state(2); y=in_state(3); b=in_state(4)
    !gx=in_polar(1); gb=in_polar(2); xb=in_polar(3);
    ! energies
    E_x = delta_E + delta_0/2.0
    E_y = delta_E - delta_0/2.0
    E_b = 2.*delta_E - delta_B
    ! pulse parameters
    tau1_new = sqrt((alpha**2. / tau1**2.) + tau1**2. )
    a_chirp = alpha / (alpha**2. + tau1**4.)
    w_1 = e_energy1 / HBAR
    w_2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    pol1y = sqrt(1-pol1x**2.)
    pol2y = sqrt(1-pol2x**2.)

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_x 
    ! actually phi/t. if this is time dependent (i.e. modulated frequency and using a time dependent rot. frame), move inside the loop
    phi = E_x / HBAR
    energies(1)=E_x; energies(2)=E_y; energies(3)=E_b
    do i = 0, n_steps - 2
        ! take first rk4 step:
        t = t_0 + i * dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omx = pol1x * laser1 + pol2x * laser2
        omy = pol1y * laser1 + pol2y * laser2
        call biex_linear_rf(out_states(i+1,:), out_polars(i+1,:), phidot, omx, omy, energies, k1r, k1i)

        ! now t -> t+h/2
        t = t_0 + i * dt + 0.5*dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omx = pol1x * laser1 + pol2x * laser2
        omy = pol1y * laser1 + pol2y * laser2
        
        call biex_linear_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, omx, omy, energies, k2r, k2i)
        call biex_linear_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot,  omx, omy, energies, k3r, k3i)
        ! now t -> t+h
        t = t_0 + i * dt + dt
        laser1 = e01 * exp(-0.5 * (t/tau1_new)**2.) * exp(-ii*(w_1-phi + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau1 * tau1_new)
        laser2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*((w_2-phi)*(t-t02) + phase))/ sqrt(2.*pi*tau2 * tau2)
        omx = pol1x * laser1 + pol2x * laser2
        omy = pol1y * laser1 + pol2y * laser2
        call biex_linear_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, omx, omy, energies, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)

end subroutine biex_linear_twopulse

subroutine biex_arbitrary_field(dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, &
    e0, delta_b, delta_E)
    ! ===============================
    ! solves biexciton / three level system for two pulses with given parameters
    ! rotating frame with energy of E_x is used. 
    ! the first pulse can be chirped by specifying alpha
    ! if big detunings are used, think about using one of those as rot. frame frequencies
    ! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: delta_E
    ! times are given in fs, energies in meV
    ! e01, e02 are pulse areas
    Real*8,intent(in) :: dt, delta_B
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state(3)
    Complex*16,intent(in):: in_polar(3), e0(2*n_steps-1)
    Real*8,intent(out):: out_state(3)
    Complex*16,intent(out):: out_polar(3)
    Real*8,intent(out):: out_states(n_steps,3)
    Complex*16,intent(out):: out_polars(n_steps,3)
    Real*8 :: HBAR
    Real*8 :: E_x, E_b
    Real*8 :: pi, phidot, phi !, t
    Real*8 :: k1r(3), k2r(3), k3r(3), k4r(3), energies(2)
    Complex*16 :: k1i(3), k2i(3), k3i(3), k4i(3), e_field
    ! constants
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    out_state = in_state
    
    !g=in_state(1); x=in_state(2); b=in_state(3)
    !gx=in_polar(1); gb=in_polar(2); xb=in_polar(3);
    ! energies
    E_x = delta_E
    E_b = 2.*delta_E - delta_B
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    ! starting parameters
    out_states(1,:) = in_state
    out_polars(1,:) = in_polar
    out_polar = in_polar

    ! phidot in meV. actually this is phidot*HBAR
    ! remember to substract phi (not phidot) in the exponential function of the electric field
    phidot = E_x 
    ! actually phi/t. if this is time dependent (i.e. modulated frequency and using a time dependent rot. frame), move inside the loop
    phi = E_x / HBAR
    energies(1)=E_x; energies(2)=E_b
    do i = 0, n_steps - 2
        ! take first rk4 step:
        ! t = t_0 + i * dt
        e_field = e0(2*i+1) ! careful: loop starts at i=0, but array indexing starts at 1
        call biex_eq_rf(out_states(i+1,:), out_polars(i+1,:), phidot, e_field, energies, k1r, k1i)

        ! now t -> t+h/2
        ! t = t_0 + i * dt + 0.5*dt
        e_field = e0(2*i+2)
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k1r, out_polars(i+1,:)+0.5*dt*k1i, phidot, e_field, energies, k2r, k2i)
        call biex_eq_rf(out_states(i+1,:)+0.5*dt*k2r, out_polars(i+1,:)+0.5*dt*k2i, phidot, e_field, energies, k3r, k3i)
        ! now t -> t+h
        ! t = t_0 + i * dt + dt
        e_field = e0(2*i+3)
        call biex_eq_rf(out_states(i+1,:)+dt*k3r, out_polars(i+1,:)+dt*k3i, phidot, e_field, energies, k4r, k4i)
        out_states(i+2,:) = out_states(i+1,:) + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        out_polars(i+2,:) = out_polars(i+1,:) + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
    end do
    out_state = out_states(n_steps,:)
    out_polar = out_polars(n_steps,:)
end subroutine biex_arbitrary_field


subroutine biex_eq_rf(in_state, in_polar, phidot, omega_x, energies, out_state, out_polar)
    implicit none
    ! equations for biexciton system in a constant rotating frame. 
    ! electric fields can be complex to include a phase
    ! or an oscillation
    Real*8,intent(in) :: in_state(3)
    Complex*16,intent(in) :: in_polar(3), omega_x
    Real*8,intent(in) :: energies(2), phidot
    Real*8,intent(out) :: out_state(3)
    Complex*16,intent(out) :: out_polar(3)
    Real*8 :: HBAR, E_x, E_b
    Complex*16 :: ii = (0.0,1.0)
    Real*8 :: g, x, b
    Complex*16 :: gx, gb, xb
    
    HBAR = 6.582119514E02  ! meV fs
    E_x=energies(1); E_b=energies(2)
    g=in_state(1); x=in_state(2); b=in_state(3)
    gx=in_polar(1); gb=in_polar(2); xb=in_polar(3)

    out_state(1) = real(0.5*ii*gx*conjg(omega_x) - 0.5*ii*omega_x*conjg(gx))
    out_polar(1) = 0.5*ii*g*omega_x + 0.5*ii*gb*conjg(omega_x) + ii*gx*(-E_x + phidot)/hbar - 0.5*ii*omega_x*x
    out_polar(2) = ii*gb*(-E_b + 2*phidot)/hbar + 0.5*ii*gx*omega_x - 0.5*ii*omega_x*xb

    out_state(2) = real(-0.5*ii*gx*conjg(omega_x) + 0.5*ii*omega_x*conjg(gx) - 0.5*ii*omega_x*conjg(xb) + 0.5*ii*xb*conjg(omega_x))
    out_polar(3) = -0.5*ii*b*omega_x - 0.5*ii*gb*conjg(omega_x) + 0.5*ii*omega_x*x + ii*xb*(-E_b + E_x + phidot)/hbar

    out_state(3) = real(0.5*ii*omega_x*conjg(xb) - 0.5*ii*xb*conjg(omega_x))
end subroutine biex_eq_rf

subroutine biex_linear_rf(in_state, in_polar, phidot, omx, omy, energies, out_state, out_polar)
    implicit none
    ! equations for biexciton system in a constant rotating frame.
    ! electric fields can be complex to include a phase
    ! or an oscillation
    Real*8,intent(in) :: in_state(4)
    Complex*16,intent(in) :: in_polar(6), omx, omy
    Real*8,intent(in) :: energies(3), phidot
    Real*8,intent(out) :: out_state(4)
    Complex*16,intent(out) :: out_polar(6)
    Real*8 :: HBAR, E_x, E_b, E_y
    Complex*16 :: ii = (0.0,1.0)
    Real*8 :: g, x, b, y
    Complex*16 :: gx, gy, gb, xy, xb, yb

    HBAR = 0.6582119514  ! meV ps
    E_x=energies(1); E_y=energies(2); E_b=energies(3)
    g=in_state(1); x=in_state(2); y=in_state(3); b=in_state(4)
    gx=in_polar(1); gy=in_polar(2); gb=in_polar(3); xy=in_polar(4); xb=in_polar(5); yb=in_polar(6)

    out_state(1) = real(0.5*ii*gx*conjg(omx) + 0.5*ii*gy*conjg(omy) - 0.5*ii*omx*conjg(gx) - 0.5*ii*omy*conjg(gy))
    out_polar(1) = 0.5*ii*g*omx + 0.5*ii*gb*conjg(omx) + ii*gx*(-E_X + phidot)/hbar - 0.5*ii*omx*x - 0.5*ii*omy*conjg(xy)
    out_polar(2) = 0.5*ii*g*omy + 0.5*ii*gb*conjg(omy) + ii*gy*(-E_Y + phidot)/hbar - 0.5*ii*omx*xy - 0.5*ii*omy*y
    out_polar(3) = ii*gb*(-E_B + 2*phidot)/hbar + 0.5*ii*gx*omx - 0.5*ii*gy*omy - 0.5*ii*omx*xb - 0.5*ii*omy*yb
    
    out_state(2) = real(-0.5*ii*gx*conjg(omx) + 0.5*ii*omx*conjg(gx) - 0.5*ii*omx*conjg(xb) + 0.5*ii*xb*conjg(omx))
    out_polar(4) = -0.5*ii*gy*conjg(omx) - 0.5*ii*omx*conjg(yb) + 0.5*ii*omy*conjg(gx) + 0.5*ii*xb*conjg(omy) &
                   + ii*xy*(E_X - E_Y)/hbar
    out_polar(5) = -0.5*ii*b*omx - 0.5*ii*gb*conjg(omx) + 0.5*ii*omx*x - 0.5*ii*omy*xy + ii*xb*(-E_B + E_X + phidot)/hbar
    
    out_state(3) = real(-0.5*ii*gy*conjg(omy) + 0.5*ii*omy*conjg(gy) + 0.5*ii*omy*conjg(yb) + 0.5*ii*yb*conjg(omy))
    out_polar(6) = 0.5*ii*b*omy - 0.5*ii*gb*conjg(omy) + 0.5*ii*omx*conjg(xy) - 0.5*ii*omy*y &
                   + ii*yb*(-E_B + E_Y + phidot)/hbar
    
    out_state(4) = real(0.5*ii*omx*conjg(xb) - 0.5*ii*omy*conjg(yb) - 0.5*ii*xb*conjg(omx) - 0.5*ii*yb*conjg(omy))
end subroutine biex_linear_rf
