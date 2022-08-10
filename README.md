# Quant-Risk-Dynamics
Implied default intensity via CDS spread curve


1. Input a cds spread curve and use bootstrapping algorithm to calculate survival prob and default intensity(hazard rates), which is the input of poisson process
2. Having a term structure from the CDS Spread
3. Test and validate a constant-lambda inputted model of homogeneous poisson process
   3.1 First method: Testing Nt(Number of events for each simulation) and having a close-to-poisson distribution
   3.2 Second method: Picking any two subintervals from each simulation, the number of events between them should have same close-to-poisson distribution
4. Generate and simulate inhomogeneous poisson process from the term structure of hazard rates from CDS
5. Validate them as step 3
