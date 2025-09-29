Canonical Record Verification

This record is cryptographically sealed.

# Verify manifest signature
gpg --verify canonical_record_manifest.asc canonical_record_manifest

# Verify PDF integrity
sha256sum -c canonical_record_manifest


Expected SHA-256 (PDF):
e286bcc1b6dbc2dc8661fb449aad66c3538b809426848cc61320649e6d33a56c

If the hash or signature fails, the record has been altered.