You are helping complete SOFE 4820U Modelling and Simulation – Tutorial 7 on Hamiltonian Monte Carlo (HMC).

Your task is to produce:
1. A complete Python solution that answers all Tutorial 7 questions.
2. A complete LaTeX report that includes the written answers, figures, and the Python code as a listing.
3. A clean project structure so I can compile and submit it easily.

Use a writing style appropriate for a 3rd- or 4th-year engineering student: clear, direct, and not overly polished or AI-sounding.

--------------------------------------------------
CONTEXT / TUTORIAL QUESTIONS
--------------------------------------------------

The tutorial questions are:

1. Explain why Metropolis-Hastings (MH) suffers from high autocorrelation in high-dimensional problems, and how Hamiltonian Monte Carlo (HMC) addresses this issue.

2. Using the provided HMC code for a 1D Gaussian distribution:
   - Run the HMC sampler with step size ε = 0.25
   - Plot the histogram of samples
   - Compare the histogram with the true normal distribution
   - Explain your observation

3. Modify the previous experiment by increasing the step size to ε = 0.8.
   - Run the sampler
   - Observe the acceptance behavior
   - Explain the effect of large step size on HMC performance

4. Consider a 2D Gaussian distribution with strong correlation.
   - Explain why MH struggles to sample from this distribution
   - Explain how HMC trajectories improve sampling efficiency

5. Why is Leapfrog integration preferred in HMC, and why is reversibility important?

The supporting slide deck says, in substance:
- MH behaves like a random walk with small local moves, so consecutive samples are highly correlated, especially in high dimensions.
- HMC uses gradient information and follows informed trajectories that allow longer moves with higher acceptance.
- HMC augments the parameter vector with momentum variables and uses Hamiltonian dynamics.
- Leapfrog updates momentum half-step, position full-step, then momentum half-step.
- Momentum flipping is used for reversibility.

--------------------------------------------------
DELIVERABLES REQUIRED
--------------------------------------------------

Create the following files:

1. tutorial7_hmc.py
2. tutorial7_report.tex
3. figures/eps_025_hist.png
4. figures/eps_08_hist.png
5. figures/optional_2d_demo.png   (only if helpful)
6. README.md

--------------------------------------------------
PYTHON REQUIREMENTS
--------------------------------------------------

Write a fully runnable Python script named tutorial7_hmc.py.

Use only standard scientific Python libraries:
- numpy
- matplotlib
- scipy.stats if needed

The script must:
1. Implement a simple 1D HMC sampler for a normal target distribution N(0,1).
2. Include helper functions such as:
   - normal_pdf(x, mu, sigma)
   - neg_log_prob(x, mu, sigma)
   - grad_U(q, mu, sigma)
   - hmc_sampler(...)

3. Use a fixed random seed for reproducibility.
4. Generate enough samples to make the histograms meaningful.
5. Track and report acceptance rate for each experiment.
6. Run two experiments:
   - step_size = 0.25
   - step_size = 0.8
7. Save histogram figures comparing:
   - normalized histogram of HMC samples
   - true N(0,1) density curve
8. Print concise numerical summaries to console:
   - acceptance rate
   - sample mean
   - sample variance
9. Include comments explaining the main parts of the code.

Important:
- Make the code robust and internally consistent.
- Compute the gradient properly.
- If the original slide code seems mathematically incomplete or simplified, correct it and implement a proper educational HMC version while keeping it simple.
- Use leapfrog integration correctly.
- Use a reasonable number of leapfrog steps and path length.
- Include burn-in handling.
- Do not overcomplicate the implementation.

Also include a short optional section in the script that demonstrates a strongly correlated 2D Gaussian:
- It does not need to fully implement 2D HMC unless that is easy.
- At minimum, generate a contour plot or brief visual that helps explain why MH random-walk proposals struggle on elongated correlated densities.
- If you do implement a simple MH demo or 2D HMC visualization, keep it compact and educational.

--------------------------------------------------
LATEX REPORT REQUIREMENTS
--------------------------------------------------

Write tutorial7_report.tex as a complete compilable LaTeX document.

Use this overall style:
- \documentclass[conference]{IEEEtran}
- Include packages needed for:
  - math
  - graphics
  - code listings
  - color
  - float placement if needed

Suggested packages:
- graphicx
- amsmath
- amssymb
- booktabs
- listings
- xcolor
- float

Author block:
- Peter Marchut
- SOFE 4820U – Modelling and Simulation
- Ontario Tech University
- Winter 2026

Title:
Tutorial 7 – Hamiltonian Monte Carlo

The report should contain:
1. Title
2. Short introduction
3. Separate section for each question
4. Figures embedded in the relevant sections
5. Acceptance-rate discussion for both experiments
6. A concise conclusion
7. An appendix or final section containing the Python code as a listing

Use \lstset for Python formatting.

--------------------------------------------------
CONTENT EXPECTATIONS FOR EACH QUESTION
--------------------------------------------------

Question 1:
Write a concise but solid conceptual explanation:
- MH makes local random-walk proposals
- In high dimensions, good proposals become hard to design
- Consecutive states stay similar, causing high autocorrelation
- Exploration becomes slow
- HMC uses gradients and momentum to move along probability contours
- This reduces random walk behavior and improves mixing

Question 2:
For ε = 0.25:
- Run the sampler
- Show histogram and true density
- Comment that the sampled histogram should generally match the true N(0,1) shape reasonably well
- Note acceptance behavior and explain why moderate step size tends to work well

Question 3:
For ε = 0.8:
- Run the sampler
- Comment on reduced acceptance and/or poorer approximation if observed
- Explain that larger step size causes larger integration error in leapfrog simulation
- This makes Hamiltonian conservation worse and reduces performance

Question 4:
For strongly correlated 2D Gaussian:
- Explain that MH random-walk proposals tend to move inefficiently across narrow, tilted contours
- Many proposals are too small and exploration is slow, or too large and they get rejected
- HMC follows trajectories informed by gradients, so it can move farther along the high-density valley efficiently

Question 5:
Explain why leapfrog is preferred:
- numerically stable
- approximately energy-preserving
- time-reversible
- volume-preserving
These properties make the Metropolis correction valid and keep acceptance rates reasonable.

Also explain reversibility:
- it helps maintain detailed balance
- it ensures the Markov chain targets the correct stationary distribution

--------------------------------------------------
WRITING / FORMATTING CONSTRAINTS
--------------------------------------------------

Important constraints:
- Do not invent fake numerical results.
- Actually compute the experimental outputs from the Python code.
- Use the real measured acceptance rates from the simulation.
- Keep explanations readable and student-like.
- Do not use overly grand claims.
- Do not use first-person language excessively.
- Avoid saying “this proves”; prefer “this suggests”, “this shows”, or “this indicates”.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Provide your answer as:
1. A short summary of what you created
2. Full contents of tutorial7_hmc.py
3. Full contents of tutorial7_report.tex
4. Full contents of README.md

If useful, also provide:
- a suggested folder structure
- compile instructions for LaTeX
- run instructions for Python

Make sure the LaTeX document references the generated figure filenames correctly and compiles cleanly.