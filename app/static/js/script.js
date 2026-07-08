/**
 * script.js
 * ==========
 * Handles the prediction form: validates on the client (basic UX
 * sanity check only — the REAL validation happens server-side via
 * Pydantic, since client-side checks can always be bypassed), calls
 * POST /predict, and renders the result or a clear error.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const btn = document.getElementById("predictBtn");
  const btnLabel = btn.querySelector(".vg-btn-ignite-label");
  const btnSpinner = btn.querySelector(".vg-btn-ignite-spinner");
  const resultCard = document.getElementById("resultCard");
  const resultValue = document.getElementById("resultValue");
  const errorBanner = document.getElementById("errorBanner");

  // Same-origin API base — works whether the frontend is served BY
  // this Flask app (templates/) or hosted separately, as long as
  // CORS is enabled on the backend (it is — see app/app.py).
  const API_BASE = window.location.origin;

  function setLoading(isLoading) {
    btn.disabled = isLoading;
    btnLabel.classList.toggle("d-none", isLoading);
    btnSpinner.classList.toggle("d-none", !isLoading);
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("d-none");
    resultCard.classList.add("d-none");
  }

  function hideError() {
    errorBanner.classList.add("d-none");
  }

  function showResult(price) {
    hideError();
    resultValue.textContent = Number(price).toFixed(2);
    resultCard.classList.remove("d-none");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function buildPayload(formData) {
    return {
      brand: formData.get("brand").trim(),
      vehicle_age: Number(formData.get("vehicle_age")),
      km_driven: Number(formData.get("km_driven")),
      seller_type: formData.get("seller_type"),
      fuel_type: formData.get("fuel_type"),
      transmission_type: formData.get("transmission_type"),
      mileage: Number(formData.get("mileage")),
      engine: Number(formData.get("engine")),
      max_power: Number(formData.get("max_power")),
      seats: Number(formData.get("seats")),
    };
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideError();

    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }

    const payload = buildPayload(new FormData(form));
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        // Surface Pydantic's field-level errors (422) clearly, or the
        // generic error message for anything else (400/500/503).
        if (data.details && Array.isArray(data.details)) {
          const fieldErrors = data.details
            .map((d) => `${d.field}: ${d.message}`)
            .join(" · ");
          showError(`Please check your inputs — ${fieldErrors}`);
        } else {
          showError(data.error || "Something went wrong. Please try again.");
        }
        return;
      }

      showResult(data.predicted_price);
    } catch (err) {
      showError("Could not reach the prediction service. Please try again.");
    } finally {
      setLoading(false);
    }
  });
});
