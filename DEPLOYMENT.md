# Live demo deployment with Vercel

Deploy this repository as two Vercel projects: the FastAPI backend from
`backend/` and the Next.js frontend from `frontend/`. The backend loads the
bundled Excel workbooks at startup.

## Keep secrets out of GitHub

- Revoke any Sarvam API key that has been shared or exposed, and create a
  replacement.
- Do not commit `.env` or `.env.local` files. The root `.gitignore` excludes
  them; the `.env.example` files contain placeholders only.
- Add the replacement API key only in the backend Vercel project's
  Environment Variables settings.
- If an API key was ever committed, revoke it; removing it from a later commit
  does not make the old key safe.

## 1. Deploy the FastAPI backend

1. Push the project to GitHub without any `.env` files.
2. In Vercel, import the GitHub repository as a new project named, for example,
   `dukansaathi-api`.
3. Set **Root Directory** to `backend`.
4. Vercel detects `app.py`, FastAPI, and `requirements.txt`. Leave the
   framework/build settings on automatic defaults.
5. Add these Environment Variables for **Production** (and Preview if wanted):
   - `SARVAM_API_KEY`: your replacement Sarvam key
   - `SARVAM_STT_MODEL`: `saaras:v4`
   - `SARVAM_STT_LANGUAGE`: `unknown`
   - `SARVAM_STT_MODE`: `transcribe`
   - `SARVAM_TTS_MODEL`: `bulbul:v4-flash`
   - `SARVAM_TTS_SPEAKER`: `shubh_enhi_ads`
   - `SARVAM_TRANSLATION_MODEL`: `sarvam-translate:v1`
   - `FRONTEND_ORIGINS`: temporarily `http://localhost:3000`
6. Deploy the project and copy its production URL, for example
   `https://dukansaathi-api.vercel.app`.
7. Open `https://YOUR-BACKEND-URL/status`. The response should show both
   datasets connected with non-zero record counts.

## 2. Deploy the Next.js frontend

1. Import the same GitHub repository again as a second Vercel project, for
   example `dukansaathi-demo`.
2. Set **Root Directory** to `frontend`.
3. Add the environment variable
   `NEXT_PUBLIC_API_BASE_URL=https://YOUR-BACKEND-URL` with the actual backend
   URL and no trailing slash.
4. Deploy and copy the production frontend URL.
5. In the backend Vercel project, change `FRONTEND_ORIGINS` to that exact
   frontend URL, such as `https://dukansaathi-demo.vercel.app`, then redeploy
   the backend. Multiple exact origins can be comma-separated.

## 3. Test the live demo

- Open the frontend URL and confirm the transaction and inventory cards show
  record counts.
- Try a typed question, then try a voice question and allow microphone access.
- Microphone access requires HTTPS; Vercel provides it automatically.
- Vercel's Hobby plan currently allows function durations up to 300 seconds.
  The backend calls Sarvam several times in one voice request, so test a real
  voice question before presenting and check Vercel function logs if it fails.
- Check Vercel's current plan, usage, and account requirements before
  deploying; platform limits and terms can change.

## Local environment files

For local development, copy `backend/.env.example` to `backend/.env` and add
your own key. Copy `frontend/.env.example` to the ignored
`frontend/.env.local`; use `http://localhost:8000` as the API URL locally.