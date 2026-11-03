/** GET /api/params -> the generation parameters the engine actually implements. */
import { PARAMS, DEFAULTS } from "../lib/params.mjs";
import { json } from "../lib/api.mjs";

export default function handler(_req, res) {
  json(res, 200, { params: PARAMS, defaults: DEFAULTS });
}
