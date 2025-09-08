namespace shapes {
    inline float circle(float2 uv, float2 center, float radius) {
        return length(uv - center) - radius;
    }
}