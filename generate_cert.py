"""
Generate Self-Signed SSL Certificate
=====================================
Creates a self-signed certificate for local HTTPS development.
"""

from OpenSSL import crypto
import socket

# Get local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print(f"Generating certificate for:")
print(f"  - localhost")
print(f"  - 127.0.0.1")
print(f"  - {local_ip}")
print(f"  - {hostname}")

# Create a key pair
key = crypto.PKey()
key.generate_key(crypto.TYPE_RSA, 2048)

# Create a self-signed cert
cert = crypto.X509()
cert.get_subject().C = "PH"
cert.get_subject().ST = "Metro Manila"
cert.get_subject().L = "Manila"
cert.get_subject().O = "InterviewAI"
cert.get_subject().OU = "Development"
cert.get_subject().CN = local_ip

# Add Subject Alternative Names (SANs) for all addresses
san_list = [
    f"DNS:localhost",
    f"DNS:{hostname}",
    f"IP:127.0.0.1",
    f"IP:{local_ip}"
]
san_extension = crypto.X509Extension(
    b"subjectAltName",
    False,
    ", ".join(san_list).encode()
)
cert.add_extensions([san_extension])

cert.set_serial_number(1000)
cert.gmtime_adj_notBefore(0)
cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
cert.set_issuer(cert.get_subject())
cert.set_pubkey(key)
cert.sign(key, 'sha256')

# Save certificate and key
with open("cert.pem", "wb") as f:
    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

with open("key.pem", "wb") as f:
    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

print("\n✅ Certificate generated successfully!")
print("   - cert.pem (certificate)")
print("   - key.pem (private key)")
print("\n⚠️  Note: Browsers will show a security warning for self-signed certs.")
print("   Click 'Advanced' → 'Proceed to site' to continue.")
