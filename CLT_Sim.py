"""
Interactive Dashboard: Central Limit Theorem Demo
Demonstrates how sampling distributions approach normality regardless of population distribution
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Central Limit Theorem Explorer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎲 Central Limit Theorem Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Watch how the magic of statistics transforms any distribution into a normal distribution!</p>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Distribution selection
distribution_type = st.sidebar.selectbox(
    "Choose Population Distribution",
    ["Uniform", "Exponential", "Binomial", "Poisson", "Beta", "Chi-Square"],
    index=0
)

# Sample parameters
sample_size = st.sidebar.slider(
    "Sample Size (n)",
    min_value=2,
    max_value=200,
    value=30,
    step=1,
    help="Number of observations in each sample"
)

num_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Number of sample means to generate"
)

# Distribution-specific parameters
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Distribution Parameters")

if distribution_type == "Uniform":
    low = st.sidebar.slider("Lower Bound", -10.0, 0.0, 0.0, 0.5)
    high = st.sidebar.slider("Upper Bound", 1.0, 20.0, 10.0, 0.5)
    population = np.random.uniform(low, high, 100000)
    dist_params = f"Uniform({low}, {high})"
    
elif distribution_type == "Exponential":
    scale = st.sidebar.slider("Scale (λ⁻¹)", 0.1, 5.0, 1.0, 0.1)
    population = np.random.exponential(scale, 100000)
    dist_params = f"Exponential(λ={1/scale:.2f})"
    
elif distribution_type == "Binomial":
    n_trials = st.sidebar.slider("Number of Trials (n)", 1, 50, 10, 1)
    p = st.sidebar.slider("Success Probability (p)", 0.0, 1.0, 0.5, 0.05)
    population = np.random.binomial(n_trials, p, 100000)
    dist_params = f"Binomial(n={n_trials}, p={p})"
    
elif distribution_type == "Poisson":
    lam = st.sidebar.slider("Lambda (λ)", 0.5, 20.0, 5.0, 0.5)
    population = np.random.poisson(lam, 100000)
    dist_params = f"Poisson(λ={lam})"
    
elif distribution_type == "Beta":
    alpha = st.sidebar.slider("Alpha (α)", 0.5, 10.0, 2.0, 0.5)
    beta = st.sidebar.slider("Beta (β)", 0.5, 10.0, 5.0, 0.5)
    population = np.random.beta(alpha, beta, 100000)
    dist_params = f"Beta(α={alpha}, β={beta})"
    
else:  # Chi-Square
    df = st.sidebar.slider("Degrees of Freedom", 1, 20, 5, 1)
    population = np.random.chisquare(df, 100000)
    dist_params = f"Chi-Square(df={df})"

# Generate sampling distribution
np.random.seed(42)
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population, size=sample_size, replace=True)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# Calculate statistics
pop_mean = np.mean(population)
pop_std = np.std(population)
sample_mean_mean = np.mean(sample_means)
sample_mean_std = np.std(sample_means)
theoretical_std = pop_std / np.sqrt(sample_size)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Population Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot population distribution
    ax1.hist(population, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(pop_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {pop_mean:.3f}')
    ax1.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Original Population: {dist_params}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    st.pyplot(fig1)
    
    st.info(f"""
    **Population Statistics:**
    - Mean (μ): {pop_mean:.4f}
    - Standard Deviation (σ): {pop_std:.4f}
    - Distribution: {distribution_type}
    """)

with col2:
    st.subheader("🎯 Sampling Distribution of the Mean")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Plot sampling distribution
    ax2.hist(sample_means, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black', label='Sample Means')
    
    # Overlay normal curve
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    normal_curve = stats.norm.pdf(x, sample_mean_mean, sample_mean_std)
    ax2.plot(x, normal_curve, 'g-', linewidth=3, label='Normal Fit')
    
    ax2.axvline(sample_mean_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {sample_mean_mean:.3f}')
    ax2.set_xlabel('Sample Mean', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of {num_samples} Sample Means (n={sample_size})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    
    st.success(f"""
    **Sampling Distribution Statistics:**
    - Mean of Sample Means: {sample_mean_mean:.4f}
    - Std Dev of Sample Means: {sample_mean_std:.4f}
    - Theoretical Std Error (σ/√n): {theoretical_std:.4f}
    - Difference: {abs(sample_mean_std - theoretical_std):.4f}
    """)

# Statistical comparison
st.markdown("---")
st.subheader("📊 Statistical Analysis")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric(
        label="Population Mean (μ)",
        value=f"{pop_mean:.4f}",
        delta=None
    )

with col4:
    st.metric(
        label="Sample Mean Estimate",
        value=f"{sample_mean_mean:.4f}",
        delta=f"{(sample_mean_mean - pop_mean):.4f}"
    )

with col5:
    error_reduction = ((pop_std - sample_mean_std) / pop_std) * 100
    st.metric(
        label="Variability Reduction",
        value=f"{error_reduction:.1f}%",
        delta=f"↓ {sample_mean_std:.4f}"
    )

# Normality test
st.markdown("---")
st.subheader("🔍 Testing for Normality")

col6, col7 = st.columns([1, 1])

with col6:
    # Shapiro-Wilk test (for smaller samples)
    if num_samples <= 5000:
        _, p_value_sw = stats.shapiro(sample_means)
        st.write("**Shapiro-Wilk Test:**")
        st.write(f"- P-value: {p_value_sw:.6f}")
        if p_value_sw > 0.05:
            st.write("- ✅ **Result:** Cannot reject normality (p > 0.05)")
        else:
            st.write("- ⚠️ **Result:** Evidence against normality (p < 0.05)")
    else:
        st.write("**Shapiro-Wilk Test:**")
        st.write("- Sample too large (using alternative tests)")
    
    # Kolmogorov-Smirnov test
    _, p_value_ks = stats.kstest(sample_means, 'norm', args=(sample_mean_mean, sample_mean_std))
    st.write("\n**Kolmogorov-Smirnov Test:**")
    st.write(f"- P-value: {p_value_ks:.6f}")
    if p_value_ks > 0.05:
        st.write("- ✅ **Result:** Consistent with normal distribution (p > 0.05)")
    else:
        st.write("- ⚠️ **Result:** Some deviation from normality (p < 0.05)")

with col7:
    # Q-Q plot
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    stats.probplot(sample_means, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: Checking Normality', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

# Educational content
st.markdown("---")
st.subheader("📚 Understanding the Central Limit Theorem")

with st.expander("🤔 What is the Central Limit Theorem?"):
    st.markdown("""
    The **Central Limit Theorem (CLT)** is one of the most important concepts in statistics. It states that:
    
    > When you take sufficiently large random samples from **any** population and calculate the mean of each sample, 
    > the distribution of these sample means will approximate a **normal distribution**, regardless of the population's 
    > original shape.
    
    **Key Points:**
    - Works for ANY population distribution (uniform, exponential, binomial, etc.)
    - The approximation improves as sample size increases
    - The mean of sample means equals the population mean (μ)
    - The standard deviation of sample means = σ/√n (standard error)
    """)

with st.expander("🎯 Why is this important?"):
    st.markdown("""
    The CLT is the foundation for:
    
    1. **Hypothesis Testing**: Allows us to make inferences about population parameters
    2. **Confidence Intervals**: Enables us to estimate population means with known precision
    3. **Quality Control**: Helps monitor processes even when individual measurements aren't normal
    4. **Survey Sampling**: Justifies using sample statistics to estimate population parameters
    5. **Machine Learning**: Underpins many algorithms and validation techniques
    """)

with st.expander("🔬 Try These Experiments"):
    st.markdown("""
    **Experiment 1: Effect of Sample Size**
    - Keep the distribution the same
    - Increase sample size from 2 to 200
    - Notice how the sampling distribution becomes more "bell-shaped"
    
    **Experiment 2: Different Population Distributions**
    - Try highly skewed distributions (Exponential, Chi-Square with low df)
    - Watch how even very non-normal populations produce normal sampling distributions
    
    **Experiment 3: Standard Error Validation**
    - Compare the "Std Dev of Sample Means" with "Theoretical Std Error (σ/√n)"
    - They should be very close, confirming CLT predictions!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Interactive Statistical Learning Dashboard</strong></p>
    <p>Built with Streamlit • Data generated using NumPy and SciPy</p>
    <p>📧 Questions? Explore the code on GitHub!</p>
</div>
""", unsafe_allow_html=True)