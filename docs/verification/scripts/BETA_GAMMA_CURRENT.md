# The photon-coupling bound on current CMB data (downtime work)

`verify_beta_gamma_current.py` (output `verify_beta_gamma_current_output.txt`, inputs `beta_gamma_current_inputs.csv`) runs the model and
method of `verify_beta_gamma.py` unchanged on the current public CMB fits: the ACT DR6 LCDM table (arXiv:2503.14452), columns ACT, Planck,
W-ACT and P-ACT, every input of a row from one fit and checked against the paper's text before running. The combinations with BAO are left
out (late-time data IAM modifies); SPT-3G D1 (arXiv:2506.20707) is left out because its table reports the angle under another definition with
no matching sound horizon or last-scattering redshift.

**Finding.** The simplified model (sound horizon fixed, massless neutrinos) does not reproduce each fit's own measured angle at beta_g = 0:
it sits +1.1 sigma (Planck) to +3.5 sigma (ACT alone) high. A profile bound from a baseline that already sits high is tighter than the data
warrant, so the profile bounds (ACT alone 0.0026; P-ACT 0.0039) and the book's 0.0052 (from a baseline +1.4 sigma high) depend on that
miss. The bound from the shift alone (theta_* at beta_g against the model's own theta_* at 0, within 2 of the fit's errors) does not:
Planck 0.0099, W-ACT 0.0093, ACT alone 0.0100, **P-ACT 0.0080 (beta_g/beta_m < 0.051)**. The photon exemption holds either way: the matter
coupling beta_m on photon paths shifts the angle by about 30 sigma at Planck precision (`verify_beta_gamma_output.txt`). The book printed
0.0052, 3.3 % of beta_m and 'at least 30 times more weakly', from the profile bound; it now quotes the shift-only bound on P-ACT,
0.0080, 5.1 % of beta_m, at least 19 times (`verify_beta_gamma.py` prints it first, so every book check reads it; the Planck 2018 profile
stays in that output, labelled, and in the figure as the dashed curve). A full
refit with the coupling inside a Boltzmann solver would replace both.
