! compile with: f2py -c tls.f95 -m tls
subroutine tls(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, e_energy, e0, delta_e, tau)
! ===============================
! solves two level system for t = t_0,...,t_0+dt*(n_steps-1)
! so for example, n_steps = 3: t = t_0, t_0+dt, t_0+2*dt
! tau, e_energy, e0 are pulse parameters
! uses RF with energy difference of the two levels
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, tau
    Real*8,intent(in) :: dt, e_energy, e0, delta_e
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, pi, w_start, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    w_start = e_energy / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    ! because we use a rotating frame (RF) with the energy difference, the
    ! delta term is always zero.
    phidot = delta_e/HBAR
    delta = phidot - delta_e/HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR)*t)/ sqrt(2.*pi*tau*tau)
        
        call tls_eq(g, p, omm, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR)*t)/ sqrt(2.*pi*tau*tau)
        
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR)*t)/ sqrt(2.*pi*tau*tau)
        
        call tls_eq(g+k3r*dt, p+k3i*dt, omm, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls

subroutine tls_chirp(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, tau_0,&
    e_energy, alpha, e0, delta_e)
! ===============================
! solves two level system for one pulse with given parameters
! tau_0, e_energy, alpha, e0 are pulse parameters
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0
    Real*8,intent(in) :: dt, tau_0, e_energy, alpha, e0, delta_e
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, tau, a_chirp, pi, w_start, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    tau = sqrt((alpha**2. / tau_0**2.) + tau_0**2. )
    a_chirp = alpha / (alpha**2. + tau_0**4.)
    w_start = e_energy / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    phidot = delta_e/HBAR
    delta = phidot - delta_e/HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g, p, omm, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        call tls_eq(g+k3r*dt, p+k3i*dt, omm, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_chirp

subroutine tls_chirp_a(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, tau_0,&
    e_energy, a_chirp, e0, delta_e)
! ===============================
! solves two level system for one pulse with given parameters
! tau_0, e_energy, alpha, e0 are pulse parameters
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0
    Real*8,intent(in) :: dt, tau_0, e_energy, a_chirp, e0, delta_e
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, tau, pi, w_start, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    tau = tau_0
    w_start = e_energy / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    phidot = delta_e/HBAR
    delta = phidot - delta_e/HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g, p, omm, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start-delta_e/HBAR + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        
        call tls_eq(g+k3r*dt, p+k3i*dt, omm, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_chirp_a

subroutine tls_fm(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, tau_0,&
    detuning, detuning_small, e0, fm_freq)
! ===============================
! solves two level system for one frequency modulated pulse with given parameters
! tau_0, e0 (pulsearea), detuning, detuning_small are pulse parameters
! here, only a constant frequency modulation of fm_freq is supported.
! the laser frequency is then f(t) = detuning + detuning_small * sin(fm_freq*t)
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0
    Real*8,intent(in) :: dt, tau_0, detuning, detuning_small, e0, fm_freq
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Complex*16,intent(out):: out_polars(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, tau, pi, w_start, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    tau = tau_0
    w_start = detuning / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g; out_polars(1)=p;
    ! delta = phidot
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        ! omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        phidot = w_start + (detuning_small/HBAR) * sin(fm_freq*t)
        delta = phidot ! - delta_e/HBAR
        call tls_eq(g, p, omm, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        ! omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        phidot = w_start + (detuning_small/HBAR) * sin(fm_freq*t)
        delta = phidot ! - delta_e/HBAR
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        ! omm = e0 * exp(-0.5 * (t/tau)**2.) * exp(-ii*(w_start + 0.5*a_chirp*t)*t)/ sqrt(2.*pi*tau * tau_0)
        phidot = w_start + (detuning_small/HBAR) * sin(fm_freq*t)
        delta = phidot ! - delta_e/HBAR
        call tls_eq(g+k3r*dt, p+k3i*dt, omm, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
        out_polars(i+2)=p;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_fm

subroutine tls_twopulse(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, tau1,&
    tau2, e_energy1, e_energy2, a_chirp1, a_chirp2, e01, e02, rf_energy, t02, phase)
! ===============================
! solves two level system for two pulses with given parameters
! tau_0, e_energy, alpha, e0, t02 are pulse parameters
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, t02
    Real*8,intent(in) :: dt, tau1, e_energy1, a_chirp1, e01, rf_energy, tau2, e_energy2, a_chirp2, e02, phase
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Complex*16,intent(out):: out_polars(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, pi, w_start1, w_start2, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm1, omm2, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    w_start1 = e_energy1 / HBAR
    w_start2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    out_polars(1)=p;
    ! rotating frame with system energy: delta = 0, phidot = 0
    ! so the 'delta' in the bloch eq. is zero, and the fields include the whole detuning
    ! different: rf with first detuning: delta = e_energy1/HBAR = phidot
    ! so the first laser has no oscillation at all, and the detuning is included in the bloch eq.
    phidot = rf_energy/HBAR
    delta = rf_energy/HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-phidot &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*( (w_start2-phidot &
               + 0.5*a_chirp2*(t-t02))*(t-t02) + phase ))/ sqrt(2.*pi*tau2 * tau2)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g, p, omm1+omm2, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-phidot &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*( (w_start2-phidot &
               + 0.5*a_chirp2*(t-t02))*(t-t02) + phase ))/ sqrt(2.*pi*tau2 * tau2)
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm1+omm2, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm1+omm2, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-phidot &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 * exp(-0.5 * ((t-t02)/tau2)**2.) * exp(-ii*( (w_start2-phidot &
               + 0.5*a_chirp2*(t-t02))*(t-t02) + phase ))/ sqrt(2.*pi*tau2 * tau2)
        
        call tls_eq(g+k3r*dt, p+k3i*dt, omm1+omm2, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
        out_polars(i+2)=p;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_twopulse

subroutine tls_twopulse_cw_second(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars, tau1,&
    e_energy1, e_energy2, a_chirp1, a_chirp2, e01, e02, delta_e, t02, slope)
! ===============================
! solves two level system for two pulses with given parameters
! tau_0, e_energy, alpha, e0, t02 are pulse parameters
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, t02
    Real*8,intent(in) :: dt, tau1, e_energy1, a_chirp1, e01, delta_e, e_energy2, a_chirp2, e02, slope
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Complex*16,intent(out):: out_polars(n_steps)
    Real*8 :: HBAR
    Real*8 :: t, pi, w_start1, w_start2, phidot
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, omm1, omm2, ii
    ! constants
    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! pulse parameters
    w_start1 = e_energy1 / HBAR
    w_start2 = e_energy2 / HBAR
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    out_polars(1)=p;
    phidot = delta_e/HBAR
    delta = phidot - delta_e/HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-delta_e/HBAR &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 /(1 + exp(-slope*(t-t02)) ) * exp(-ii*(w_start2-delta_e/HBAR &
               + 0.5*a_chirp2*(t-t02))*(t-t02))
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g, p, omm1+omm2, delta, k1r, k1i)
        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-delta_e/HBAR &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 /(1 + exp(-slope*(t-t02)) ) * exp(-ii*(w_start2-delta_e/HBAR &
               + 0.5*a_chirp2*(t-t02))*(t-t02))
        !phidot = w_start + a_chirp * t
        !delta = phidot - delta_e/HBAR
        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, omm1+omm2, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, omm1+omm2, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        omm1 = e01 * exp(-0.5 * (t/tau1)**2.) * exp(-ii*(w_start1-delta_e/HBAR &
               + 0.5*a_chirp1*t)*t)/ sqrt(2.*pi*tau1 * tau1)
        omm2 = e02 /(1 + exp(-slope*(t-t02)) ) * exp(-ii*(w_start2-delta_e/HBAR &
               + 0.5*a_chirp2*(t-t02))*(t-t02))
        call tls_eq(g+k3r*dt, p+k3i*dt, omm1+omm2, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
        out_polars(i+2)=p;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_twopulse_cw_second

subroutine tls_fm_rectangular(t_0, tau, dt, n_steps, amplitude, det1, det2, omega_mod, in_state, in_polar, out_states, out_polars)
    !
    !
    !
    !
    implicit none
    integer,intent(in) :: n_steps
    real*8,intent(in) :: tau, dt, amplitude, det1, det2, omega_mod, in_state
    complex*16,intent(in) :: in_polar
    real*8,intent(out) :: out_states(n_steps)
    complex*16,intent(out) :: out_polars(n_steps)
    real*8 :: pi, HBAR, g, t, t_0
    complex*16 :: ii, p, pulse
    real*8 :: delta, k1r, k2r, k3r, k4r, factor
    complex*16 :: k1i, k2i, k3i, k4i
    integer :: i=0

    ii = (0.0,1.0)
    pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 

    ! pulse parameters
    !w_start1 = e_energy1 / HBAR
    !w_start2 = e_energy2 / HBAR
    do i = 0, n_steps - 2

        ! take rk4 step
        t = t_0 + i * dt
        ! pulse stuff
        pulse = 0
        if ( abs(t) < tau/2.0 ) then
            pulse = amplitude
        end if
        factor = 1.0
        if ( sin(omega_mod*t) < 0 ) then
            factor = -1.0
        end if
        delta = (det1 + det2 * factor)/HBAR

        call tls_eq(g, p, pulse, delta, k1r, k1i)

        ! t -> t+h/2 
        t = t_0 + i*dt + 0.5*dt
        ! pulse stuff
        pulse = 0
        if ( abs(t) < tau/2.0 ) then
            pulse = amplitude
        end if
        factor = 1.0
        if ( sin(omega_mod*t) < 0 ) then
            factor = -1.0
        end if
        delta = (det1 + det2 * factor)/HBAR

        call tls_eq(g+k1r*0.5*dt, p+k1i*0.5*dt, pulse, delta, k2r, k2i)
        call tls_eq(g+k2r*0.5*dt, p+k2i*0.5*dt, pulse, delta, k3r, k3i)
        ! t -> t+h
        t = t_0 + i*dt + dt
        ! pulse stuff
        pulse = 0
        if ( abs(t) < tau/2.0 ) then
            pulse = amplitude
        end if
        factor = 1.0
        if ( sin(omega_mod*t) < 0 ) then
            factor = -1.0
        end if
        delta = (det1 + det2 * factor)/HBAR

        call tls_eq(g+k3r*dt, p+k3i*dt, pulse, delta, k4r, k4i)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
        out_polars(i+2)=p;
    end do
end subroutine tls_fm_rectangular

subroutine tls_arbitrary_field(t_0, dt, n_steps, in_state, in_polar, out_state, out_polar, out_states, out_polars,&
           e0, delta_e, gamma_x)
! ===============================
! solves two level system for arbitrary electric field given as array
! first in array has to be E-field for t0, then t0+dt/2, then t0+dt
! uses RF with energy difference of the two levels
! ===============================
    implicit none
    integer :: i=0
    Real*8,intent(in) :: t_0, dt, delta_e, gamma_x
    integer,intent(in) :: n_steps
    Real*8,intent(in):: in_state
    Complex*16,intent(in):: in_polar, e0(2*n_steps-1)
    Real*8,intent(out):: out_state
    Complex*16,intent(out):: out_polar
    Real*8,intent(out):: out_states(n_steps)
    Complex*16,intent(out):: out_polars(n_steps)
    Real*8 :: HBAR
    Real*8 :: t
    Real*8 :: g, delta, k1r, k2r, k3r, k4r
    Complex*16 :: p, k1i, k2i, k3i, k4i, e_field
    ! constants
    !ii = (0.0,1.0)
    !pi = 4.0d0*atan(1.0d0)
    HBAR = 6.582119514E02  ! meV fs
    ! starting parameters
    g=in_state;
    p=in_polar; 
    ! steps for the loop 
    !n_steps = int(abs(t_end - t_0)/dt) - 1

    out_states(1)=g;
    out_polars(1)=p;
    !phidot = delta_e/HBAR
    delta = delta_e !phidot - delta_e/HBAR
    !tau1 = 3000
    !e01 = pi
    do i = 0, n_steps - 2
        ! take rk4 step
        t = t_0 + i * dt
        e_field = e0(2*i+1)  ! careful: loop starts at i=0, but array indexing starts at 1
        !e_field = e01 * exp(-0.5 * (t/tau1)**2.) / sqrt(2.*pi*tau1 * tau1)
        call tls_eq_lindblad(g, p, e_field, delta, k1r, k1i, gamma_x)
        ! t -> t+h/2 
        t = t_0 + i * dt + 0.5*dt
        e_field = e0(2*i+2)
        !e_field = e01 * exp(-0.5 * (t/tau1)**2.) / sqrt(2.*pi*tau1 * tau1)
        !omm = e0 * exp(-0.5 * (t/tau)**2.)/ sqrt(2.*pi*tau * tau_0)
        call tls_eq_lindblad(g+k1r*0.5*dt, p+k1i*0.5*dt, e_field, delta, k2r, k2i, gamma_x)
        call tls_eq_lindblad(g+k2r*0.5*dt, p+k2i*0.5*dt, e_field, delta, k3r, k3i, gamma_x)
        ! t -> t+h
        t = t_0 + i * dt + dt
        e_field = e0(2*i+3)
        !e_field = e01 * exp(-0.5 * (t/tau1)**2.) / sqrt(2.*pi*tau1 * tau1)
        call tls_eq_lindblad(g+k3r*dt, p+k3i*dt, e_field, delta, k4r, k4i, gamma_x)
        ! update parameters
        g=g + (1.0/6.0)*dt*(k1r + 2.0*k2r + 2.0*k3r + k4r);
        p=p + (1.0/6.0)*dt*(k1i + 2.0*k2i + 2.0*k3i + k4i);
        out_states(i+2)=g;
        out_polars(i+2)=p;
    end do
    out_state=g;
    out_polar=p;
end subroutine tls_arbitrary_field

subroutine tls_eq(in_state, in_polar, e_f, delta, out_state, out_polar)
    implicit none
    Real*8,intent(in) :: in_state, delta
    Complex*16,intent(in) :: in_polar, e_f
    Real*8,intent(out) :: out_state
    Complex*16,intent(out) :: out_polar
    Complex*16 :: ii = (0.0,1.0)

    out_state = AIMAG(conjg(e_f)*in_polar)
    ! delta = phidot - omega_0, where omega_0 = system frequency
    out_polar = ii*delta*in_polar + ii*0.5*e_f*(1-2*in_state)
end subroutine tls_eq

subroutine tls_eq_lindblad(in_state, in_polar, e_f, delta, out_state, out_polar, gamma_x)
    implicit none
    Real*8,intent(in) :: in_state, delta, gamma_x
    Complex*16,intent(in) :: in_polar, e_f
    Real*8,intent(out) :: out_state
    Complex*16,intent(out) :: out_polar
    Complex*16 :: ii = (0.0,1.0)

    out_state = AIMAG(conjg(e_f)*in_polar) - gamma_x*in_state
    ! delta = phidot - omega_0, where omega_0 = system frequency
    out_polar = (ii*delta - 0.5*gamma_x)*in_polar + ii*0.5*e_f*(1-2*in_state)
end subroutine tls_eq_lindblad
