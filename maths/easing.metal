#ifndef MATHS_EASING_METAL
#define MATHS_EASING_METAL

/// Clamp input to [0,1] to ensure valid easing domain
inline float clamp01(float t) {
    return clamp(t, 0.0f, 1.0f);
}

/// Generic power easing (Quadratic, Cubic, Quartic, Quintic)
inline float pow_ease(float t, float exp) {
    return pow(clamp01(t), exp);
}

/// Generic exponential easing
inline float expo_ease(float t, float base) {
    return (pow(base, clamp01(t)) - 1.0f) / (base - 1.0f);
}

/// Circular easing (easeIn circle = sqrt(1 - t^2))
inline float circ_ease_in(float t) {
    t = clamp01(t);
    return 1.0f - sqrt(1.0f - t * t);
}

/// Circular easing out
inline float circ_ease_out(float t) {
    t = clamp01(t) - 1.0f;
    return sqrt(1.0f - t * t);
}

/// Circular easing in/out
inline float circ_ease_in_out(float t) {
    t = clamp01(t) * 2.0f;
    if (t < 1.0f) return 0.5f * (1.0f - sqrt(1.0f - t * t));
    t -= 2.0f;
    return 0.5f * (sqrt(1.0f - t * t) + 1.0f);
}

/// Sinusoidal easing
inline float sine_ease_in(float t) {
    return 1.0f - cos(clamp01(t) * M_PI_F * 0.5f);
}
inline float sine_ease_out(float t) {
    return sin(clamp01(t) * M_PI_F * 0.5f);
}
inline float sine_ease_in_out(float t) {
    return 0.5f * (1.0f - cos(clamp01(t) * M_PI_F));
}

/// Polynomial family (Quadratic, Cubic, Quartic, Quintic) via exponent
inline float ease_in(float t, float exp) {
    return pow_ease(t, exp);
}
inline float ease_out(float t, float exp) {
    return 1.0f - pow_ease(1.0f - t, exp);
}
inline float ease_in_out(float t, float exp) {
    t = clamp01(t) * 2.0f;
    if (t < 1.0f) return 0.5f * pow_ease(t, exp);
    return 1.0f - 0.5f * pow_ease(2.0f - t, exp);
}

/// Exponential family
inline float expo_ease_in(float t, float base) {
    return expo_ease(t, base);
}
inline float expo_ease_out(float t, float base) {
    return 1.0f - expo_ease(1.0f - t, base);
}
inline float expo_ease_in_out(float t, float base) {
    t = clamp01(t);
    if (t < 0.5f) return 0.5f * expo_ease(t * 2.0f, base);
    return 1.0f - 0.5f * expo_ease((1.0f - t) * 2.0f, base);
}

/// Physically-inspired spring easing.
/// Returns oscillatory easing with overshoot/settling.
/// Inputs:
/// - stiffness (k): spring constant
/// - damping   (d): damping factor
/// - mass      (m): mass of object
inline float spring_ease(float t, float stiffness, float damping, float mass) {
    t = clamp01(t);

    float omega0 = sqrt(stiffness / mass);   // natural frequency
    float zeta   = damping / (2.0f * sqrt(stiffness * mass)); // damping ratio

    if (zeta < 1.0f) {
        // Underdamped (oscillatory)
        float omega_d = omega0 * sqrt(1.0f - zeta * zeta);
        return 1.0f - exp(-zeta * omega0 * t) *
                     (cos(omega_d * t) + (zeta / sqrt(1.0f - zeta * zeta)) * sin(omega_d * t));
    } else if (zeta == 1.0f) {
        // Critically damped
        return 1.0f - exp(-omega0 * t) * (1.0f + omega0 * t);
    } else {
        // Overdamped
        float omega_d = omega0 * sqrt(zeta * zeta - 1.0f);
        float A = exp((-zeta * omega0 + omega_d) * t);
        float B = exp((-zeta * omega0 - omega_d) * t);
        return 1.0f - (A - B) / (2.0f * omega_d);
    }
}

#endif // MATHS_EASING_METAL
