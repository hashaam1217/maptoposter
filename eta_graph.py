#!/usr/bin/env python3
"""
Departure Time ETA Grapher

Given two points (addresses/place names or "lat, lon" coordinates), this module
queries a traffic-aware routing API for the driving ETA at regular departure-time
intervals (every 30 minutes by default) and renders a graph of drive time vs.
departure time, highlighting the best time to leave.

Traffic-aware providers (one API key required):
  - TomTom Routing API   (env: TOMTOM_API_KEY,      https://developer.tomtom.com)
  - Google Directions API (env: GOOGLE_MAPS_API_KEY, requires billing-enabled key)

A --demo mode generates synthetic traffic data so the graph can be previewed
without an API key.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import requests
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from tqdm import tqdm

ETA_GRAPHS_DIR = "eta_graphs"

REQUEST_TIMEOUT_SECONDS = 30
SECONDS_BETWEEN_REQUESTS = 0.25

TOMTOM_ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute/{locations}/json"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# Chart styling: one series -> one hue; recessive grid/axes; ink for text.
LINE_COLOR = "#2563EB"
BEST_COLOR = "#15803D"
WORST_COLOR = "#B91C1C"
GRID_COLOR = "#E5E7EB"
INK_PRIMARY = "#1F2937"
INK_SECONDARY = "#6B7280"
SURFACE = "#FFFFFF"


class RoutingError(Exception):
    """Raised when a routing provider request fails or returns no route."""


@dataclass
class EtaSample:
    """A single ETA measurement for one departure time."""

    departure: datetime
    duration_seconds: float

    @property
    def arrival(self) -> datetime:
        return self.departure + timedelta(seconds=self.duration_seconds)

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


def resolve_point(value: str) -> tuple[float, float, str]:
    """
    Resolve a user-supplied point to coordinates.

    Accepts either a "lat, lon" pair (decimal or DMS, parsed by lat_lon_parser)
    or a free-form place name/address geocoded via Nominatim.

    :param value: Coordinate pair or place name
    :return: (latitude, longitude, display label)
    """
    parts = value.split(",")
    if len(parts) == 2:
        try:
            lat, lon = parse(parts[0]), parse(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon, f"{lat:.5f}, {lon:.5f}"
        except ValueError:
            pass

    print(f"Geocoding: {value}")
    geolocator = Nominatim(user_agent="maptoposter_eta_graph")
    location = geolocator.geocode(value)
    if location is None:
        raise RoutingError(f"Could not geocode '{value}'. Try coordinates as 'lat, lon'.")
    label = location.address.split(",")[0]
    return location.latitude, location.longitude, label


def build_departures(start: datetime, hours: float, interval_minutes: int) -> list[datetime]:
    """
    Build the list of departure times to sample.

    :param start: First departure time
    :param hours: Length of the departure window in hours
    :param interval_minutes: Minutes between samples
    :return: List of timezone-aware departure datetimes
    """
    count = int(hours * 60 // interval_minutes) + 1
    return [start + timedelta(minutes=interval_minutes * i) for i in range(count)]


def fetch_eta_tomtom(origin, destination, departure: datetime, api_key: str) -> float:
    """
    Fetch a traffic-aware ETA from the TomTom Routing API.

    :param origin: (lat, lon) of the start point
    :param destination: (lat, lon) of the end point
    :param departure: Departure time (timezone-aware)
    :param api_key: TomTom API key
    :return: Travel time in seconds including traffic
    """
    locations = f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}"
    params = {
        "key": api_key,
        "departAt": departure.isoformat(timespec="seconds"),
        "traffic": "true",
        "travelMode": "car",
        "computeTravelTimeFor": "all",
    }
    response = requests.get(
        TOMTOM_ROUTING_URL.format(locations=locations), params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise RoutingError(f"TomTom API error {response.status_code}: {response.text[:300]}")
    routes = response.json().get("routes")
    if not routes:
        raise RoutingError("TomTom returned no route between these points.")
    return routes[0]["summary"]["travelTimeInSeconds"]


def fetch_eta_google(origin, destination, departure: datetime, api_key: str) -> float:
    """
    Fetch a traffic-aware ETA from the Google Directions API.

    :param origin: (lat, lon) of the start point
    :param destination: (lat, lon) of the end point
    :param departure: Departure time (timezone-aware, must be in the future)
    :param api_key: Google Maps API key
    :return: Travel time in seconds (in traffic when available)
    """
    params = {
        "key": api_key,
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "departure_time": int(departure.timestamp()),
        "mode": "driving",
    }
    response = requests.get(GOOGLE_DIRECTIONS_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RoutingError(f"Google API error {response.status_code}: {response.text[:300]}")
    data = response.json()
    if data.get("status") != "OK" or not data.get("routes"):
        raise RoutingError(f"Google Directions returned status '{data.get('status')}': "
                           f"{data.get('error_message', 'no route found')}")
    leg = data["routes"][0]["legs"][0]
    duration = leg.get("duration_in_traffic") or leg["duration"]
    return duration["value"]


def fetch_eta_demo(origin, destination, departure: datetime, api_key: str) -> float:
    """
    Generate a synthetic ETA with morning and evening rush-hour peaks.

    Used to preview the graph without an API key. Base drive time scales with
    straight-line distance between the points.
    """
    lat_km = (destination[0] - origin[0]) * 111.0
    lon_km = (destination[1] - origin[1]) * 111.0 * math.cos(math.radians(origin[0]))
    distance_km = max(math.hypot(lat_km, lon_km), 1.0)
    base_seconds = distance_km / 55.0 * 3600  # ~55 km/h average off-peak

    hour = departure.hour + departure.minute / 60.0
    morning = 0.55 * math.exp(-((hour - 8.25) ** 2) / 1.8)
    evening = 0.75 * math.exp(-((hour - 17.5) ** 2) / 2.6)
    return base_seconds * (1.0 + morning + evening)


PROVIDERS = {
    "tomtom": (fetch_eta_tomtom, "TOMTOM_API_KEY"),
    "google": (fetch_eta_google, "GOOGLE_MAPS_API_KEY"),
    "demo": (fetch_eta_demo, None),
}


def collect_samples(origin, destination, departures, provider: str, api_key: str) -> list[EtaSample]:
    """
    Query the provider for an ETA at each departure time.

    :return: List of EtaSample, one per successful departure query
    """
    fetch, _ = PROVIDERS[provider]
    samples = []
    for departure in tqdm(departures, desc=f"Fetching ETAs ({provider})", unit="req"):
        try:
            seconds = fetch(origin, destination, departure, api_key)
            samples.append(EtaSample(departure, seconds))
        except RoutingError as e:
            tqdm.write(f"  ⚠ {departure.strftime('%H:%M')}: {e}")
        if provider != "demo":
            time.sleep(SECONDS_BETWEEN_REQUESTS)
    return samples


def format_minutes(minutes: float) -> str:
    """Format a duration in minutes as 'Xh YYm' or 'YY min'."""
    if minutes >= 60:
        return f"{int(minutes // 60)}h {int(round(minutes % 60)):02d}m"
    return f"{int(round(minutes))} min"


def plot_samples(samples, origin_label, destination_label, interval_minutes, output_path):
    """
    Render the ETA vs. departure time graph and save it as a PNG.

    Highlights the best (minimum) and worst (maximum) departure times.
    """
    best = min(samples, key=lambda s: s.duration_seconds)
    worst = max(samples, key=lambda s: s.duration_seconds)
    times = [s.departure for s in samples]
    minutes = [s.duration_minutes for s in samples]

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(times, minutes, color=LINE_COLOR, linewidth=2, zorder=3)
    ax.fill_between(times, minutes, color=LINE_COLOR, alpha=0.08, zorder=1)

    for sample, color, label in ((best, BEST_COLOR, "Best"), (worst, WORST_COLOR, "Worst")):
        ax.scatter([sample.departure], [sample.duration_minutes], s=64, color=color, zorder=4,
                   edgecolors=SURFACE, linewidths=2)
        offset = 14 if sample is worst else -22
        ax.annotate(
            f"{label}: leave {sample.departure.strftime('%H:%M')}\n{format_minutes(sample.duration_minutes)}",
            xy=(sample.departure, sample.duration_minutes),
            xytext=(0, offset), textcoords="offset points",
            ha="center", va="bottom" if offset > 0 else "top",
            fontsize=10, fontweight="bold", color=color,
        )

    ax.set_title(f"Drive time by departure time\n{origin_label}  →  {destination_label}",
                 fontsize=14, fontweight="bold", color=INK_PRIMARY, loc="left", pad=14)
    ax.set_xlabel(f"Departure time ({times[0].strftime('%a %b %d')}, every {interval_minutes} min)",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Drive time (minutes)", fontsize=10, color=INK_SECONDARY)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times[0].tzinfo))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(times) // 12)))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.margins(x=0.02)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def print_summary(samples, origin_label, destination_label):
    """Print a text table of ETAs and the best/worst departure times."""
    best = min(samples, key=lambda s: s.duration_seconds)
    worst = max(samples, key=lambda s: s.duration_seconds)

    print(f"\nRoute: {origin_label} → {destination_label}")
    print(f"{'Departure':>10}  {'Drive time':>10}  {'Arrival':>8}")
    for s in samples:
        marker = "  ← best" if s is best else ("  ← worst" if s is worst else "")
        print(f"{s.departure.strftime('%a %H:%M'):>10}  {format_minutes(s.duration_minutes):>10}  "
              f"{s.arrival.strftime('%H:%M'):>8}{marker}")

    saved = worst.duration_minutes - best.duration_minutes
    print(f"\nBest time to leave:  {best.departure.strftime('%A %H:%M')} "
          f"({format_minutes(best.duration_minutes)}, arrive {best.arrival.strftime('%H:%M')})")
    print(f"Worst time to leave: {worst.departure.strftime('%A %H:%M')} "
          f"({format_minutes(worst.duration_minutes)})")
    print(f"Leaving at the best time saves up to {format_minutes(saved)}.")


def parse_start(value: str | None, interval_minutes: int) -> datetime:
    """
    Determine the first departure time.

    Defaults to now rounded up to the next interval boundary; accepts an ISO
    datetime ('2026-08-08T06:00') or a time ('06:00', meaning today/tomorrow).
    Providers reject past departure times, so times already passed roll forward
    one day.
    """
    now = datetime.now().astimezone()
    if value is None:
        minutes_past = (now.minute % interval_minutes) * 60 + now.second
        start = now + timedelta(seconds=interval_minutes * 60 - minutes_past)
        return start.replace(second=0, microsecond=0)

    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=now.tzinfo)
    except ValueError:
        try:
            t = datetime.strptime(value, "%H:%M").time()
        except ValueError:
            raise SystemExit(f"Could not parse --start '{value}'. Use ISO format or HH:MM.")
        parsed = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if parsed <= now:
        parsed += timedelta(days=1)
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Graph driving ETA between two points at regular departure intervals "
                    "to find the best time to leave.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--from", dest="origin", required=True,
                        help="Start point: place name/address or 'lat, lon'")
    parser.add_argument("--to", dest="destination", required=True,
                        help="End point: place name/address or 'lat, lon'")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between departure samples")
    parser.add_argument("--hours", type=float, default=24, help="Length of the departure window in hours")
    parser.add_argument("--start", default=None,
                        help="First departure ('2026-08-08T06:00' or '06:00'); default: next interval from now")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                        help="Routing provider; default: auto-detect from available API keys")
    parser.add_argument("--output", "-o", default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.interval < 1:
        raise SystemExit("--interval must be at least 1 minute.")

    provider = args.provider
    if provider is None:
        for name, (_, env_var) in PROVIDERS.items():
            if env_var and os.environ.get(env_var):
                provider = name
                break
        else:
            raise SystemExit(
                "No API key found. Set TOMTOM_API_KEY (free key: https://developer.tomtom.com) or "
                "GOOGLE_MAPS_API_KEY, or run with '--provider demo' to preview with synthetic traffic."
            )

    _, env_var = PROVIDERS[provider]
    api_key = os.environ.get(env_var) if env_var else None
    if env_var and not api_key:
        raise SystemExit(f"Provider '{provider}' requires the {env_var} environment variable.")

    try:
        origin = resolve_point(args.origin)
        destination = resolve_point(args.destination)
    except RoutingError as e:
        raise SystemExit(str(e))

    start = parse_start(args.start, args.interval)
    departures = build_departures(start, args.hours, args.interval)
    print(f"Sampling {len(departures)} departure times from "
          f"{start.strftime('%a %b %d %H:%M')} every {args.interval} min ({provider}).")

    samples = collect_samples(origin[:2], destination[:2], departures, provider, api_key)
    if not samples:
        raise SystemExit("No ETAs could be fetched — check the points and your API key.")

    print_summary(samples, origin[2], destination[2])

    output_path = args.output
    if output_path is None:
        os.makedirs(ETA_GRAPHS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(ETA_GRAPHS_DIR, f"eta_{stamp}.png")
    plot_samples(samples, origin[2], destination[2], args.interval, output_path)
    print(f"\nGraph saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
