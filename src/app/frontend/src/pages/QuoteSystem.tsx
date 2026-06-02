import { useEffect, useState } from 'react';
import { Car, ShieldCheck, Loader2, Check, Lock } from 'lucide-react';
import { api } from '../lib/api';

const JOHN = 'POL-MOTOR-00000001';

// Consumer-facing motor insurance quote page. Looks like a real insurer's
// online quote journey — a proper form, then a clean price. Deliberately
// shows ONLY what a customer would see (price, cover, instalments) — no
// internal rating breakdown, model outputs, or behaviour scores.
export default function QuoteSystem() {
  const [form, setForm]       = useState<any>(null);
  const [quote, setQuote]     = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    api.livePricingPolicy(JOHN).then(p => setForm({
      reg:        'MA18 KZR',
      make:       p.vehicle.make,
      model:      p.vehicle.model,
      year:       p.vehicle.year,
      value:      p.vehicle.value,
      mileage:    p.vehicle.mileage,
      parking:    p.vehicle.parking,
      age:        p.driver.age,
      licence:    p.driver.license_years,
      occupation: p.driver.occupation,
      postcode:   p.driver.postcode_area,
      ncd:        p.driver.no_claims_years,
      cover:      'Comprehensive',
      excess:     250,
    })).catch(() => setForm(null));
  }, []);

  const getQuote = async () => {
    if (!form) return;
    setLoading(true); setError(null);
    // The 6 editable fields are pushed as live inputs; the rest are pulled
    // from the feature store server-side.
    const overrides: Record<string, any> = {
      annual_mileage:   Number(form.mileage) || 0,
      vehicle_value:    Number(form.value) || 0,
      driver_age:       Number(form.age) || 0,
      no_claims_years:  Number(form.ncd) || 0,
      parking_overnight: form.parking,
      occupation_class:  form.occupation,
    };
    try {
      const r = await api.livePricingQuoteWhatIf(JOHN, overrides);
      if (!r.ok || r.status_code !== 200) {
        setError(r.detail || 'Sorry — we couldn\'t price this right now. Please try again.');
      } else {
        setQuote(r);
      }
    } catch {
      setError('Sorry — we couldn\'t price this right now. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const annual  = quote?.result?.final_premium;
  const monthly = annual != null ? annual / 12 : null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Brand bar */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
            <Car className="w-5 h-5 text-white" />
          </div>
          <div className="font-bold text-lg tracking-tight">Bricksurance<span className="text-blue-600"> Motor</span></div>
          <div className="ml-auto text-xs text-slate-500 flex items-center gap-1.5">
            <Lock className="w-3.5 h-3.5" /> Secure quote
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold mb-1">Your car insurance quote</h1>
        <p className="text-slate-500 text-sm mb-6">Check your details and get your price. It only takes a moment.</p>

        {!form ? (
          <div className="text-slate-400 text-sm">Loading…</div>
        ) : (
          <div className="grid md:grid-cols-[1fr_360px] gap-6 items-start">
            {/* Form */}
            <div className="space-y-5">
              <FormCard title="Your car">
                <Field label="Registration"><Input value={form.reg} onChange={v => setForm({ ...form, reg: v })} mono /></Field>
                <Field label="Make & model"><Input value={`${form.make} ${form.model}`} readOnly /></Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Year"><Input value={String(form.year)} readOnly /></Field>
                  <Field label="Estimated value (£)">
                    <Input value={String(form.value)} onChange={v => setForm({ ...form, value: v.replace(/[^0-9]/g, '') })} />
                  </Field>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Annual mileage">
                    <Input value={String(form.mileage)} onChange={v => setForm({ ...form, mileage: v.replace(/[^0-9]/g, '') })} />
                  </Field>
                  <Field label="Overnight parking">
                    <Select value={form.parking} onChange={v => setForm({ ...form, parking: v })}
                            options={['Driveway', 'Garage', 'On road', 'Car park']} />
                  </Field>
                </div>
              </FormCard>

              <FormCard title="About you">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Age"><Input value={String(form.age)} onChange={v => setForm({ ...form, age: v.replace(/[^0-9]/g, '') })} /></Field>
                  <Field label="Years licence held"><Input value={String(form.licence)} readOnly /></Field>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Occupation">
                    <Select value={form.occupation} onChange={v => setForm({ ...form, occupation: v })}
                            options={['Student', 'Professional', 'Manual', 'Clerical', 'Retired', 'Self-employed', 'Unemployed']} />
                  </Field>
                  <Field label="Postcode"><Input value={form.postcode} mono readOnly /></Field>
                </div>
                <Field label="No-claims discount">
                  <Select value={String(form.ncd)} onChange={v => setForm({ ...form, ncd: v.replace(/[^0-9]/g, '') })}
                          options={['0', '1', '2', '3', '4', '5']} />
                </Field>
              </FormCard>

              <FormCard title="Your cover">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Cover level">
                    <Select value={form.cover} onChange={v => setForm({ ...form, cover: v })}
                            options={['Comprehensive', 'Third party, fire & theft', 'Third party only']} />
                  </Field>
                  <Field label="Voluntary excess">
                    <Select value={`£${form.excess}`} onChange={() => {}}
                            options={['£0', '£100', '£250', '£500']} />
                  </Field>
                </div>
                <label className="flex items-center gap-2 text-sm text-slate-600 mt-1">
                  <span className="w-9 h-5 rounded-full bg-blue-600 relative">
                    <span className="absolute right-0.5 top-0.5 w-4 h-4 rounded-full bg-white" />
                  </span>
                  Smart Miles telematics black box <span className="text-blue-600 font-medium">included</span>
                </label>
              </FormCard>
            </div>

            {/* Quote panel */}
            <aside className="md:sticky md:top-8">
              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                <div className="bg-blue-600 text-white px-5 py-3 text-sm font-semibold">
                  {form.cover} cover
                </div>
                <div className="p-5">
                  {annual != null ? (
                    <>
                      <div className="text-slate-500 text-sm">Annual price</div>
                      <div className="text-4xl font-bold tracking-tight">£{Math.round(annual).toLocaleString()}</div>
                      <div className="text-sm text-slate-500 mt-1 mb-4">
                        or 12 monthly payments of <span className="font-semibold text-slate-700">£{monthly!.toFixed(2)}</span> · 0% APR
                      </div>
                      <ul className="space-y-1.5 text-sm text-slate-600 border-t border-slate-100 pt-4 mb-5">
                        <Inc text="Smart Miles telematics black box" />
                        <Inc text="Courtesy car as standard" />
                        <Inc text="Windscreen & glass cover" />
                        <Inc text="Uninsured driver promise" />
                      </ul>
                      <button disabled
                              className="w-full py-3 rounded-xl bg-blue-600/40 text-white font-semibold cursor-not-allowed mb-2">
                        Continue to purchase
                      </button>
                      <button onClick={getQuote} disabled={loading}
                              className="w-full py-2 text-sm text-blue-600 hover:text-blue-800 font-medium inline-flex items-center justify-center gap-1.5">
                        {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                        Recalculate price
                      </button>
                    </>
                  ) : (
                    <div className="text-center py-6">
                      <ShieldCheck className="w-10 h-10 text-blue-600 mx-auto mb-3" />
                      <p className="text-sm text-slate-600 mb-5">Get your personalised comprehensive quote.</p>
                      <button onClick={getQuote} disabled={loading}
                              className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold inline-flex items-center justify-center gap-2">
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                        Get my quote
                      </button>
                    </div>
                  )}
                  {error && <div className="mt-4 text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg p-3">{error}</div>}
                </div>
              </div>
              <p className="text-[11px] text-slate-400 mt-3 px-1">
                Quote valid for 30 days. Price based on the details provided and your telematics driving data.
              </p>
              {quote?.inputs_pushed != null && (
                <p className="text-[11px] text-slate-400 mt-1 px-1">
                  Priced live from <span className="text-slate-500">{quote.features_pulled} features pulled</span> from the
                  feature store + <span className="text-slate-500">{quote.inputs_pushed} inputs you set</span>.
                </p>
              )}
            </aside>
          </div>
        )}
      </main>

      {/* Subtle internal return link */}
      <footer className="border-t border-slate-200 py-4">
        <div className="max-w-5xl mx-auto px-6 flex items-center justify-between text-[11px] text-slate-400">
          <span>© Bricksurance SE · Demo environment</span>
          <a href="/" className="hover:text-slate-600 underline-offset-2 hover:underline">Pricing Workbench</a>
        </div>
      </footer>
    </div>
  );
}

function FormCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
      <div className="text-sm font-semibold text-slate-800 mb-3">{title}</div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-[11px] uppercase tracking-wide text-slate-400 font-medium">{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );
}
function Input({ value, onChange, readOnly, mono }: { value: string; onChange?: (v: string) => void; readOnly?: boolean; mono?: boolean }) {
  return (
    <input value={value} readOnly={readOnly || !onChange}
           onChange={e => onChange?.(e.target.value)}
           className={`w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 ${readOnly || !onChange ? 'bg-slate-50 text-slate-600' : 'bg-white'} ${mono ? 'font-mono' : ''}`} />
  );
}
function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  const has = options.includes(value);
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500">
      {!has && <option>{value}</option>}
      {options.map(o => <option key={o}>{o}</option>)}
    </select>
  );
}
function Inc({ text }: { text: string }) {
  return (
    <li className="flex items-center gap-2">
      <Check className="w-4 h-4 text-emerald-600 shrink-0" /> {text}
    </li>
  );
}
