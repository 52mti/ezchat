const TOKEN_KEY = "license_admin_access_token";

let accessToken = sessionStorage.getItem(TOKEN_KEY) || "";

export function setAccessToken(token) {
  accessToken = token || "";
  if (accessToken) {
    sessionStorage.setItem(TOKEN_KEY, accessToken);
  } else {
    sessionStorage.removeItem(TOKEN_KEY);
  }
}

export function clearAccessToken() {
  setAccessToken("");
}

async function readBody(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function errorMessage(data, fallback) {
  if (typeof data.detail === "string" && data.detail) {
    return data.detail;
  }
  return fallback;
}

export async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body !== undefined && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }
  let response = await fetch(path, { ...options, headers, credentials: "include" });
  if (response.status === 401 && path !== "/api/admin/login" && path !== "/api/admin/refresh") {
    const refreshed = await refreshSession();
    if (refreshed) {
      headers.Authorization = `Bearer ${accessToken}`;
      response = await fetch(path, { ...options, headers, credentials: "include" });
    }
  }
  const data = await readBody(response);
  if (!response.ok) {
    const error = new Error(errorMessage(data, "请求失败"));
    error.status = response.status;
    throw error;
  }
  return data;
}

export async function refreshSession() {
  const response = await fetch("/api/admin/refresh", { method: "POST", credentials: "include" });
  const data = await readBody(response);
  if (!response.ok) {
    clearAccessToken();
    return false;
  }
  setAccessToken(data.access_token);
  return true;
}

export async function ensureSession() {
  if (!accessToken) {
    return refreshSession();
  }
  try {
    await api("/api/admin/me");
    return true;
  } catch {
    return false;
  }
}

export function login(username, password) {
  return api("/api/admin/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }).then((data) => {
    setAccessToken(data.access_token);
    return data;
  });
}

export function logout() {
  return api("/api/admin/logout", { method: "POST" }).finally(() => {
    clearAccessToken();
  });
}
