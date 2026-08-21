// White-label brand config.
//
// Defaults committed here are the public demo identity (Bricksurance). A
// deployment can override them at BUILD time via Vite env vars, so a
// client-specific build (e.g. a sandbox handed to a carrier) carries its own
// identity WITHOUT that identity living in this (public) repo:
//
//   VITE_BRAND_NAME="AXA" VITE_BRAND_SHORT="AXA" \
//   VITE_BRAND_ACCENT="#e2231a" VITE_BRAND_KIND="prototype" npm run build
//
// KIND drives disclaimer wording:
//   'fictional' → a made-up insurer, all data synthetic (public demo)
//   'prototype' → a working prototype of the production workbench for a real
//                 carrier, running on synthetic/illustrative data (sandbox)
// import.meta.env is Vite's build-time env; cast since this project doesn't
// pull in vite/client ambient types.
const env = ((import.meta as any).env ?? {}) as Record<string, string | undefined>;

export const brand = {
  name:   env.VITE_BRAND_NAME  || 'Bricksurance SE',
  short:  env.VITE_BRAND_SHORT || 'Bricksurance',
  accent: env.VITE_BRAND_ACCENT || '#2563eb',
  kind:   (env.VITE_BRAND_KIND || 'fictional') as 'fictional' | 'prototype',
};

export const isPrototype = brand.kind === 'prototype';

// Short one-liner for footers / small print.
export const disclaimerShort = isPrototype
  ? `Prototype of the production pricing workbench — synthetic, illustrative data; not ${brand.short}'s live book.`
  : `${brand.name} is a fictional insurer; all data is synthetic.`;

// Longer "about this" paragraph.
export const disclaimerLong = isPrototype
  ? `About this prototype — a working prototype of the production pricing workbench, running on synthetic, illustrative data in a Databricks sandbox. It demonstrates the end-to-end capability (ingestion, modelling, governance, quoting) on the Databricks platform. The policies, quotes, claims and financial figures shown are illustrative and are not ${brand.short}'s real portfolio.`
  : `About this demo — ${brand.name} is a fictional insurer. The pricing models, policies, quotes, claims and financial figures are entirely synthetic and for demonstration only.`;
