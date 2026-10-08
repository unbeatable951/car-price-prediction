/**
 * script.js
 * ==========
 * Handles the prediction form:
 * 1. Fetches available car brands from GET /brands on page load and populates the dropdown.
 * 2. Validates user input before submitting.
 * 3. Calls POST /predict and renders the predicted price or friendly errors/warnings.
 */

document.addEventListener("DOMContentLoaded", () => {
  const brandSelect = document.getElementById("brand");
  const form = document.getElementById("predictForm");
  const btn = document.getElementById("predictBtn");
  const btnLabel = btn ? btn.querySelector(".vg-btn-ignite-label") : null;
  const btnSpinner = btn ? btn.querySelector(".vg-btn-ignite-spinner") : null;
  const resultCard = document.getElementById("resultCard");
  const resultValue = document.getElementById("resultValue");
  const errorBanner = document.getElementById("errorBanner");
  const warningBanner = document.getElementById("warningBanner");

  // Same-origin API base — works whether served via templates or reverse proxy
  const API_BASE = window.location.origin;

  // 1. Fetch available brands and populate select element
  async function loadBrands() {
    if (!brandSelect) return;
    try {
      const response = await fetch(`${API_BASE}/brands`);
      if (!response.ok) {
        throw new Error(`Failed to load brands (HTTP ${response.status})`);
      }
      const data = await response.json();
      if (Array.isArray(data.brands) && data.brands.length > 0) {
        brandSelect.innerHTML = '<option value="" disabled selected>Choose a brand</option>';
        data.brands.forEach((brand) => {
          const opt = document.createElement("option");
          opt.value = brand;
          opt.textContent = brand;
          brandSelect.appendChild(opt);
        });
      } else {
        brandSelect.innerHTML = '<option value="" disabled selected>No brands available</option>';
      }
    } catch (err) {
      console.error("Error loading brands:", err);
      brandSelect.innerHTML = '<option value="" disabled selected>Error loading brands (refresh page)</option>';
    }
  }

  loadBrands();

  function setLoading(isLoading) {
    if (!btn) return;
    btn.disabled = isLoading;
    if (btnLabel) btnLabel.classList.toggle("d-none", isLoading);
    if (btnSpinner) btnSpinner.classList.toggle("d-none", !isLoading);
  }

  function showError(message) {
    if (errorBanner) {
      errorBanner.textContent = message;
      errorBanner.classList.remove("d-none");
    }
    if (warningBanner) warningBanner.classList.add("d-none");
    if (resultCard) resultCard.classList.add("d-none");
  }

  function hideError() {
    if (errorBanner) errorBanner.classList.add("d-none");
  }

  function showResult(price, warning) {
    hideError();
    const numPrice = Number(price);
    const unitEl = document.getElementById("resultUnit");
    const fullValueEl = document.getElementById("resultFullValue");

    let displayPrice;
    let unitText = "Lakh";

    // The ML model outputs price in raw INR (e.g. 2532432.27 = ~25.32 Lakhs)
    if (numPrice >= 10000000) {
      displayPrice = (numPrice / 10000000).toFixed(2);
      unitText = "Cr";
    } else if (numPrice >= 100000) {
      displayPrice = (numPrice / 100000).toFixed(2);
      unitText = "Lakh";
    } else {
      displayPrice = numPrice.toFixed(2);
      unitText = "Lakh";
    }

    if (resultValue) resultValue.textContent = displayPrice;
    if (unitEl) unitEl.textContent = ` ${unitText}`;

    if (fullValueEl) {
      const totalRupees = Math.round(numPrice >= 100000 ? numPrice : numPrice * 100000);
      fullValueEl.textContent = `(₹ ${totalRupees.toLocaleString("en-IN")})`;
    }

    if (resultCard) {
      resultCard.classList.remove("d-none");
      resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    if (warningBanner) {
      if (warning) {
        warningBanner.textContent = warning;
        warningBanner.classList.remove("d-none");
      } else {
        warningBanner.classList.add("d-none");
      }
    }
  }

  function buildPayload(formData) {
    return {
      brand: formData.get("brand"),
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

  if (form) {
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

        showResult(data.predicted_price, data.warning);
      } catch (err) {
        showError("Could not reach the prediction service. Please try again.");
      } finally {
        setLoading(false);
      }
    });
  }
});