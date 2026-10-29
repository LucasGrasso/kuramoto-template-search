import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

open BigOperators Real

/-!
# Formalización de Kuramoto desde la Topología
Definiciones y demostraciones principales de los lemas topológicos,
combinatorios y perturbativos.
-/

/-- Estructura para las diferencias de fase envueltas en un ciclo de longitud k
    sin aristas antípodas, donde cada diferencia pertenece a (-π, π). -/
structure CicloSinAntipodas (k : ℕ) where
  dθ : Fin k → ℝ
  no_antipoda_inf : ∀ i, -π < dθ i
  no_antipoda_sup : ∀ i, dθ i < π

/-- Una arista está relajada (en R) si cos(Δθ) > 0, equivalente en (-π, π) a |Δθ| < π / 2. -/
def esRelajada (x : ℝ) : Prop := |x| < π / 2

/-- Una arista está en tensión (en T) si cos(Δθ) ≤ 0, equivalente en (-π, π) a π / 2 ≤ |Δθ| < π. -/
def enTension (x : ℝ) : Prop := π / 2 ≤ |x| ∧ |x| < π

/-! ## Lema 2.1: Sincronización y Número de Enrollamiento -/

/-- Lema 2.1 (⇐): Si un estado es sincronizado (Δθ = 0 en toda arista),
    el número de enrollamiento q(C) es 0. -/
theorem lema_2_1_sincronizado_implica_q_cero
    {k : ℕ} (dθ : Fin k → ℝ) (h_sinc : ∀ i, dθ i = 0) :
    (1 / (2 * π)) * (∑ i : Fin k, dθ i) = 0 := by
  have h_sum : (∑ i : Fin k, dθ i) = 0 :=
    Finset.sum_eq_zero (fun i _ => h_sinc i)
  rw [h_sum, mul_zero]

/-- Lema 2.1 (⇒): Cuando q ≡ 0 permite levantar las fases a ℝ, la condición de equilibrio
    implica ∑_e Δθ_e * sin(Δθ_e) = 0. Como Δθ_e * sin(Δθ_e) ≥ 0 en toda arista sin antípodas,
    cada término individual debe anularse. -/
theorem lema_2_1_energia_disipacion_nula
    {E_type : Type*} [Fintype E_type]
    (dθ : E_type → ℝ)
    (h_no_antipoda : ∀ e, 0 ≤ dθ e * Real.sin (dθ e))
    (h_equilibrio_global : ∑ e : E_type, dθ e * Real.sin (dθ e) = 0) :
    ∀ e, dθ e * Real.sin (dθ e) = 0 := by
  intro e
  exact (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => h_no_antipoda i)).mp
    h_equilibrio_global e (Finset.mem_univ e)

/-! ## Lema 3.1: Cubrimientos Ternarios -/

section Lema3_1

variable {V : Type*} [DecidableEq V] [Fintype V]

/-- Principio de inclusión-exclusión para 3 vecindarios cerrados en V:
    |A ∩ B ∩ C| + 2|V| ≥ |A| + |B| + |C|. -/
lemma card_inter_tres (A B C : Finset V) :
    (A ∩ B ∩ C).card + 2 * (Fintype.card V) ≥ A.card + B.card + C.card := by
  have hAB : (A ∩ B).card + (A ∪ B).card = A.card + B.card :=
    Finset.card_inter_add_card_union A B
  have hABC : ((A ∩ B) ∩ C).card + ((A ∩ B) ∪ C).card = (A ∩ B).card + C.card :=
    Finset.card_inter_add_card_union (A ∩ B) C
  have h_ub1 : (A ∪ B).card ≤ Fintype.card V := Finset.card_le_univ (A ∪ B)
  have h_ub2 : ((A ∩ B) ∪ C).card ≤ Fintype.card V := Finset.card_le_univ ((A ∩ B) ∪ C)
  omega

/-- Lema 3.1 (2): Si cada vecindario cerrado N[u_i] tiene cardinalidad ≥ μ * n,
    la intersección común satisface |N| ≥ (3μ - 2)n. -/
theorem lema_3_1_parte_2
    (n μ N1 N2 N3 N_inter : ℝ)
    (h_inter : N_inter + 2 * n ≥ N1 + N2 + N3)
    (h1 : N1 ≥ μ * n) (h2 : N2 ≥ μ * n) (h3 : N3 ≥ μ * n) :
    N_inter ≥ (3 * μ - 2) * n := by
  linarith

/-- Lema 3.1 (3, dispersión angular): Tres distancias angulares que cubren el círculo (≥ 2π)
    garantizan que al menos una es ≥ π/2 (arista en T). -/
theorem lema_3_1_dispersion_angular
    (α₁ α₂ α₃ : ℝ)
    (h_suma : α₁ + α₂ + α₃ ≥ 2 * π) :
    α₁ ≥ π / 2 ∨ α₂ ≥ π / 2 ∨ α₃ ≥ π / 2 := by
  by_contra! h
  have hpi : 0 < π := Real.pi_pos
  linarith

/-- Lema 3.1 (4): Si todas las aristas incidentes a S están en R (forzando |N| ≤ 0)
    y n > 0, entonces μ ≤ 2/3. -/
theorem lema_3_1_parte_4
    (n μ N_card : ℝ)
    (hn : 0 < n)
    (h_cota : N_card ≥ (3 * μ - 2) * n)
    (h_vacio : N_card ≤ 0) :
    μ ≤ 2 / 3 := by
  by_contra! h_mu
  have h_pos : 0 < (3 * μ - 2) * n := mul_pos (by linarith) hn
  linarith

end Lema3_1

/-! ## Lema 3.2: Longitud en Ciclos (τ > 4|q| - k) -/

/-- Lema 3.2: En un ciclo de k aristas sin antípodas con winding number q y τ aristas en T,
    se cumple τ > 4|q| - k. -/
theorem lema_3_2_longitud_en_ciclos
    (k τ q suma_abs_R suma_abs_T : ℝ)
    (h_winding : 2 * π * |q| ≤ suma_abs_R + suma_abs_T)
    (h_cota_total : suma_abs_R + suma_abs_T < (k - τ) * (π / 2) + τ * π) :
    τ > 4 * |q| - k := by
  have h1 : (4 * |q| - k - τ) * (π / 2) < 0 := by
    calc (4 * |q| - k - τ) * (π / 2)
      _ = 2 * π * |q| - ((k - τ) * (π / 2) + τ * π) := by ring
      _ ≤ (suma_abs_R + suma_abs_T) - ((k - τ) * (π / 2) + τ * π) := by linarith
      _ < 0 := by linarith
  by_contra! h_le
  have h2 : 0 ≤ (4 * |q| - k - τ) * (π / 2) :=
    mul_nonneg (by linarith) (by linarith [Real.pi_pos])
  linarith

/-! ## Sección 5: Perturbaciones de C₄ y Estabilidad -/

/-- Sección 5.3 (Ec. 10–12): Si la forma cuadrática de primer orden es semidefinida positiva
    y la suma de rigideces colectivas es nula (∑ K_j = 0), entonces K_j = 0 para todo j. -/
theorem seccion_5_3_cancelacion_primer_orden
    (K₀ K₁ K₂ K₃ : ℝ)
    (h_eq_orden_cero : K₀ + K₁ + K₂ + K₃ = 0)
    (h_psd : ∀ η₀ η₁ η₂ η₃ : ℝ,
      0 ≤ - (K₀ * (η₀ - η₁)^2 + K₁ * (η₁ - η₂)^2 + K₂ * (η₂ - η₃)^2 + K₃ * (η₃ - η₀)^2)) :
    K₀ = 0 ∧ K₁ = 0 ∧ K₂ = 0 ∧ K₃ = 0 := by
  have h1 := h_psd 1 0 0 1
  have h2 := h_psd 1 1 0 0
  have h3 := h_psd 1 0 0 0
  have h4 := h_psd 0 1 0 0
  have h5 := h_psd 0 0 1 0
  have h6 := h_psd 0 0 0 1
  refine ⟨by linarith, by linarith, by linarith, by linarith⟩

/-- Sección 5.4 (Ec. 14–16): Identidad de simetrización para la energía interna I. -/
theorem seccion_5_4_identidad_energia_interna_par
    (w s_a s_b : ℝ) :
    w * s_a * (s_b - s_a) + w * s_b * (s_a - s_b) = - (w * (s_a - s_b)^2) := by
  ring
